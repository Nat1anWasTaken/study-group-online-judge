"""Tailscale SSH transport and immutable remote job configuration."""

import json
import os
import shlex
import subprocess
from importlib.resources import files
from pathlib import PurePosixPath

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from judge.models import JudgeResult, Resources, Submission


class RemoteConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    host: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9.:-]*$")
    user: str = Field(pattern=r"^[a-z_][a-z0-9_-]*[$]?$")
    work_root: str
    repo_url: str = "https://github.com/cerulean-works/study-group-online-judge.git"
    account: str = Field(default="ACD115198", pattern=r"^[A-Za-z0-9_-]+$")
    gpu_resource: str = Field(default="gpu", pattern=r"^[A-Za-z0-9_:-]+$")
    wandb_project: str = Field(default="study-group-labs", min_length=1)
    wandb_entity: str | None = "cerulean-labs"

    @field_validator("work_root")
    @classmethod
    def absolute_work_root(cls, value: str) -> str:
        path = PurePosixPath(value)
        if not path.is_absolute() or path == PurePosixPath("/") or ".." in path.parts:
            raise ValueError("remote work root must be an absolute non-root path")
        return str(path)

    @field_validator("repo_url")
    @classmethod
    def github_repo(cls, value: str) -> str:
        return Submission.validate_repo_url(value)

    @model_validator(mode="before")
    @classmethod
    def remove_old_revision(cls, value):
        # Read existing durable jobs without honoring their former deployment pin.
        if isinstance(value, dict) and "revision" in value:
            return {key: item for key, item in value.items() if key != "revision"}
        return value

    @classmethod
    def from_environment(cls) -> RemoteConfig | None:
        if not os.environ.get("JUDGE_SSH_HOST"):
            return None
        return cls(
            host=os.environ["JUDGE_SSH_HOST"],
            user=os.environ.get("JUDGE_SSH_USER", ""),
            work_root=os.environ.get("JUDGE_REMOTE_WORK_ROOT", ""),
            repo_url=os.environ.get(
                "JUDGE_REPO_URL", cls.model_fields["repo_url"].default
            ),
            account=os.environ.get("JUDGE_SLURM_ACCOUNT", "ACD115198"),
            gpu_resource=os.environ.get("JUDGE_SLURM_GPU_RESOURCE", "gpu"),
            wandb_project=os.environ.get("WANDB_PROJECT", "study-group-labs"),
            wandb_entity=os.environ.get("WANDB_ENTITY") or "cerulean-labs",
        )


class RemoteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    submission: Submission
    resources: Resources
    config: RemoteConfig
    setup_offset: int = Field(default=0, ge=0)
    log_offset: int = Field(default=0, ge=0)

    @property
    def trusted_root(self) -> str:
        return str(
            PurePosixPath(self.config.work_root) / "jobs" / self.job_id / "judge"
        )


class RemoteSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    slurm_job_id: str | None = Field(default=None, pattern=r"^[0-9]+$")
    slurm_state: str | None = None
    setup_log: str = Field(default="", max_length=8192)
    slurm_log: str = Field(default="", max_length=8192)
    setup_offset: int = Field(default=0, ge=0)
    log_offset: int = Field(default=0, ge=0)
    logs_remaining: bool = False
    result: JudgeResult | None = None
    error: str | None = Field(default=None, max_length=8192)
    wandb_url: str | None = None
    report_pending: bool = False


class RemoteSetupError(RuntimeError):
    """The trusted bootstrap failed before a participant job could be submitted."""


class TailscaleSSH:
    def __init__(self, socket: str = "/var/run/tailscale/tailscaled.sock") -> None:
        self.socket = socket
        self._prepared: set[tuple[RemoteConfig, str]] = set()

    def command(self, config: RemoteConfig, remote_command: str) -> list[str]:
        # Options follow the destination because tailscale ssh expects it first.
        return [
            "tailscale",
            f"--socket={self.socket}",
            "ssh",
            f"{config.user}@{config.host}",
            "-T",
            "-oBatchMode=yes",
            "-oConnectTimeout=20",
            "-oServerAliveInterval=15",
            "-oServerAliveCountMax=3",
            remote_command,
        ]

    def _run(self, config: RemoteConfig, command: str, payload: str, timeout: int):
        return subprocess.run(
            self.command(config, command),
            input=payload,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )

    def prepare(self, request: RemoteRequest) -> RemoteSnapshot | None:
        config = request.config
        key = (config, request.job_id)
        if key in self._prepared:
            return None
        script = files("judge").joinpath("setup-repo.sh").read_text()
        command = shlex.join(
            [
                "env",
                f"UV_CACHE_DIR={config.work_root}/uv-cache",
                f"HF_HOME={config.work_root}/hf-cache",
                f"JUDGE_REMOTE_WORK_ROOT={config.work_root}",
                "JUDGE_SETUP_ONCE=1",
                "bash",
                "-s",
                "--",
                config.repo_url,
                "latest",
                request.trusted_root,
            ]
        )
        # A completed job may have already had its whole workspace removed.
        # Return its durable record before cloning another evaluator for it.
        record = shlex.quote(
            str(
                PurePosixPath(config.work_root)
                / "job-records"
                / f"{request.job_id}.json"
            )
        )
        command = shlex.join(
            [
                "bash",
                "-c",
                (
                    f"if [ -f {record} ] && grep -q '\"complete\": true' {record}; "
                    f"then cat -- {record}; else {command}; fi"
                ),
            ]
        )
        process = self._run(config, command, script, 1800)
        if process.returncode:
            # SSH failures can happen after the remote command began; retry safely.
            if (
                process.returncode == 255
                or process.returncode == 1
                and not process.stdout
            ):
                raise ConnectionError(
                    process.stderr.strip() or "Tailscale SSH unavailable"
                )
            raise RemoteSetupError((process.stderr or process.stdout)[-8192:])
        if process.stdout.startswith("{"):
            cached = json.loads(process.stdout)
            stored = RemoteRequest.model_validate(cached["request"])
            exclude = {"setup_offset", "log_offset"}
            if stored.model_dump(exclude=exclude) != request.model_dump(
                exclude=exclude
            ):
                raise ValueError(
                    "Remote job identity does not match its stored request"
                )
            return RemoteSnapshot.model_validate(cached["terminal"]).model_copy(
                update={
                    "setup_offset": request.setup_offset,
                    "log_offset": request.log_offset,
                }
            )
        self._prepared.add(key)
        return None

    def poll(self, request: RemoteRequest) -> RemoteSnapshot:
        try:
            cached = self.prepare(request)
            if cached is not None:
                return cached
        except RemoteSetupError as error:
            return self.report_bootstrap_failure(request, str(error))
        trusted = request.trusted_root
        command = shlex.join(
            [
                "env",
                f"PYTHONPATH={trusted}/src",
                f"UV_CACHE_DIR={request.config.work_root}/uv-cache",
                f"{trusted}/.venv/bin/python",
                "-m",
                "judge.remote_job",
            ]
        )
        process = self._run(request.config, command, request.model_dump_json(), 1800)
        if process.returncode:
            self._prepared.discard((request.config, request.job_id))
            raise ConnectionError(process.stderr[-8192:] or "Remote helper failed")
        snapshot = RemoteSnapshot.model_validate_json(process.stdout)
        if (
            snapshot.result is not None or snapshot.error
        ) and not snapshot.report_pending:
            self._prepared.discard((request.config, request.job_id))
        return snapshot

    def report_bootstrap_failure(
        self, request: RemoteRequest, error: str
    ) -> RemoteSnapshot:
        package = files("judge")
        script = package.joinpath("remote_store.py").read_text() + "\n"
        script += (
            package.joinpath("bootstrap_report.py")
            .read_text()
            .replace("from judge.remote_store import job_lock, save_record", "")
        )
        command = shlex.join(
            [
                "env",
                f"UV_CACHE_DIR={request.config.work_root}/uv-cache",
                "uv",
                "run",
                "--no-project",
                "--with",
                "wandb>=0.30.0,<0.31",
                "python",
                "-c",
                script,
            ]
        )
        process = self._run(
            request.config,
            command,
            json.dumps(
                {
                    "request": request.model_dump(mode="json"),
                    "error": f"Trusted repository setup failed: {error}"[-8192:],
                }
            ),
            1800,
        )
        if process.returncode:
            raise ConnectionError("Remote bootstrap reporting unavailable")
        return RemoteSnapshot.model_validate_json(process.stdout)
