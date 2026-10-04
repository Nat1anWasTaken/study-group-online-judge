"""One-shot remote Slurm helper. All output on stdout is a JSON snapshot."""

import json
import os
import subprocess
import sys
import time
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from judge.models import JudgeResult
from judge.remote_reporter import publish_report
from judge.remote_store import job_lock, save_record
from judge.slurm_executor import SlurmExecutor
from judge.ssh import RemoteRequest, RemoteSnapshot


def read_log(path: Path, offset: int) -> tuple[str, int, bool]:
    if not path.is_file():
        return "", offset, False
    with path.open("rb") as stream:
        stream.seek(offset)
        chunk = stream.read(8192)
        remaining = bool(stream.read(1))
    return chunk.decode(errors="replace"), offset + len(chunk), remaining


def report(request, snapshot, workspace, record) -> None:
    """Keep SDK output out of the SSH JSON response and retry failed uploads."""
    with (
        (workspace / "output" / "reporting.log").open("a") as log,
        redirect_stdout(log),
        redirect_stderr(log),
    ):
        try:
            publish_report(request, snapshot, workspace, record)
            record.pop("report_error", None)
        except Exception as error:  # noqa: BLE001 - reporting must not stop execution
            # Malformed .netrc errors can include credential tokens.
            record["report_error"] = (
                f"{type(error).__name__}: Remote W&B publication failed"
            )
            snapshot.report_pending = True
            print(record["report_error"], flush=True)


def handle(request: RemoteRequest) -> RemoteSnapshot:
    config = request.config
    workspace = Path(config.work_root) / "jobs" / request.job_id
    output = workspace / "output"
    executor = SlurmExecutor(
        trusted_root=Path(request.trusted_root),
        account=config.account,
        gpu_resource=config.gpu_resource,
    )
    identity = request.model_dump(exclude={"setup_offset", "log_offset"})
    with job_lock(workspace):
        record_path = Path(config.work_root) / "job-records" / f"{request.job_id}.json"
        old_path = workspace / "submission.json"
        existing = record_path if record_path.exists() else old_path
        record: dict = (
            json.loads(existing.read_text())
            if existing.exists()
            else {
                "request": identity,
                "state": "preparing",
                "created_at": time.time(),
            }
        )
        stored = RemoteRequest.model_validate(record["request"]).model_dump(
            exclude={"setup_offset", "log_offset"}
        )
        if stored != identity:
            raise ValueError("Remote job identity does not match its stored request")
        if record.get("terminal") and record.get("report", {}).get("complete"):
            # Small durable records survive workspace cleanup and prevent a
            # delayed reconnect from submitting a retired job a second time.
            return RemoteSnapshot.model_validate(record["terminal"]).model_copy(
                update={
                    "setup_offset": request.setup_offset,
                    "log_offset": request.log_offset,
                }
            )
        output.mkdir(parents=True, exist_ok=True)

        def persist() -> None:
            save_record(record_path, record)
            save_record(old_path, record)

        if record["state"] == "preparing":
            report(request, RemoteSnapshot(), workspace, record)
            persist()
            try:
                with (output / "setup.log").open("a") as log:
                    environment = os.environ.copy()
                    environment.update(
                        {
                            "UV_CACHE_DIR": str(Path(config.work_root) / "uv-cache"),
                            "HF_HOME": str(Path(config.work_root) / "hf-cache"),
                            "JUDGE_REMOTE_WORK_ROOT": config.work_root,
                        }
                    )
                    subprocess.run(
                        [
                            "bash",
                            str(Path(request.trusted_root) / "src/judge/setup-repo.sh"),
                            request.submission.repo_url,
                            request.submission.commit_sha,
                            str(workspace / "submission"),
                        ],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        env=environment,
                        check=True,
                        timeout=1500,
                    )
                # Validate before entering the ambiguous submission window.
                executor.build_submit_command(
                    task_id=request.submission.task_id,
                    resources=request.resources,
                    submission=workspace / "submission",
                    output_directory=output,
                    job_name=f"judge-{request.job_id}",
                )
            except Exception as error:  # noqa: BLE001  # Persist a definite setup failure.
                record.update(
                    state="failed", error=f"{type(error).__name__}: {error}"[-8192:]
                )
                persist()
            else:
                record["state"] = "submitting"
                persist()
                try:
                    slurm_job_id = executor.submit(
                        task_id=request.submission.task_id,
                        resources=request.resources,
                        submission=workspace / "submission",
                        output_directory=output,
                        job_name=f"judge-{request.job_id}",
                        hf_home=Path(config.work_root) / "hf-cache",
                        uv_cache=Path(config.work_root) / "uv-cache",
                    )
                except subprocess.CalledProcessError as error:
                    record.update(
                        state="failed", error=(error.stderr or str(error))[-8192:]
                    )
                    persist()
                else:
                    record.update(state="submitted", slurm_job_id=slurm_job_id)
                    persist()
        snapshot = RemoteSnapshot(
            slurm_job_id=record.get("slurm_job_id"),
            setup_offset=request.setup_offset,
            log_offset=request.log_offset,
        )
        if record["state"] == "submitting":
            snapshot.error = (
                "Slurm submission outcome is ambiguous; manual reconciliation required. "
                f"Check Slurm job name judge-{request.job_id} and {record_path}."
            )
        elif record["state"] == "failed":
            snapshot.error = record["error"]
        elif record.get("terminal"):
            snapshot = RemoteSnapshot.model_validate(record["terminal"]).model_copy(
                update={
                    "report_pending": False,
                    "setup_offset": request.setup_offset,
                    "log_offset": request.log_offset,
                }
            )
        elif snapshot.slurm_job_id:
            state = executor.status(snapshot.slurm_job_id)
            if state:
                snapshot.slurm_state = state.state
                if state.terminal:
                    if not state.succeeded:
                        snapshot.error = (
                            f"Slurm job ended in {state.state} (exit {state.exit_code})"
                        )
                    else:
                        try:
                            snapshot.result = JudgeResult.model_validate_json(
                                (output / "result.json").read_text()
                            )
                        except (OSError, ValueError) as error:
                            snapshot.error = (
                                f"Missing or invalid judge result: {error}"[-8192:]
                            )
        terminal = snapshot.error is not None or snapshot.result is not None
        if terminal:
            record.setdefault("finished_at", time.time())
            record["terminal"] = snapshot.model_dump(mode="json")
            persist()
        report(request, snapshot, workspace, record)
        if terminal:
            record["terminal"] = snapshot.model_dump(mode="json")
        persist()
        if (
            terminal
            and record["state"] != "submitting"
            and record.get("report", {}).get("complete")
        ):
            marker = workspace / ".finished"
            marker.touch()
            os.utime(marker, (record["finished_at"], record["finished_at"]))
        return snapshot


def main() -> None:
    request = RemoteRequest.model_validate_json(sys.stdin.read())
    print(handle(request).model_dump_json())


if __name__ == "__main__":
    main()
