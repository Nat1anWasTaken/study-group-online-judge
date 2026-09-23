import argparse
import os
import time
from collections.abc import Callable
from pathlib import Path

from judge.database import (
    claim_next_job,
    complete_job,
    fail_job,
    migrate_database,
)
from judge.executor import DockerExecutor
from judge.models import Job, JudgeResult
from judge.repository import checkout_repository, create_job_workspace
from judge.tasks import TASKS


def run_job(
    job: Job,
    *,
    database_path: Path,
    work_root: Path,
    executor: DockerExecutor,
    on_output: Callable[[str], None],
) -> Job:
    """Check out, execute, and persist one previously claimed job."""

    on_output(f"[judge] starting job {job.id}\n")
    try:
        workspace = create_job_workspace(work_root, job.id)
        submission = checkout_repository(
            job.submission.repo_url,
            job.submission.commit_sha,
            workspace / "submission",
        )
        task = TASKS[job.submission.task_id]
        output_directory = workspace / "output"
        execution = executor.run(
            task_id=task.id,
            resources=task.resources,
            submission=submission,
            output_directory=output_directory,
            on_output=on_output,
        )
        if execution.returncode != 0:
            raise RuntimeError(f"Task runner exited with status {execution.returncode}")

        result_path = output_directory / "result.json"
        result = JudgeResult.model_validate_json(result_path.read_text())
        completed = complete_job(database_path, job.id, result)
        on_output(f"[judge] completed job {job.id}\n")
        return completed
    except Exception as error:  # noqa: BLE001 - a worker must persist job failures
        message = f"{type(error).__name__}: {error}"
        failed = fail_job(database_path, job.id, message)
        on_output(f"[judge] failed job {job.id}: {message}\n")
        return failed


def run_worker(
    *,
    database_path: Path,
    work_root: Path,
    runner_image: str,
    poll_interval_seconds: float = 1,
    once: bool = False,
) -> Job | None:
    """Poll SQLite and process jobs one at a time."""

    migrate_database(database_path)
    executor = DockerExecutor(runner_image)

    while True:
        job = claim_next_job(database_path)
        if job is None:
            if once:
                return None
            time.sleep(poll_interval_seconds)
            continue

        completed = run_job(
            job,
            database_path=database_path,
            work_root=work_root,
            executor=executor,
            on_output=lambda line: print(line, end="", flush=True),
        )
        if once:
            return completed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    arguments = parser.parse_args()

    runner_image = os.environ.get("JUDGE_RUNNER_IMAGE")
    if not runner_image:
        raise RuntimeError("JUDGE_RUNNER_IMAGE must be configured")

    run_worker(
        database_path=Path(os.environ.get("JUDGE_DATABASE_PATH", "data/judge.db")),
        work_root=Path(os.environ.get("JUDGE_WORK_ROOT", "work")),
        runner_image=runner_image,
        once=arguments.once,
    )


if __name__ == "__main__":
    main()
