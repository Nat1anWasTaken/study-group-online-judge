import re
from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class JobStatus(StrEnum):
    """Lifecycle state for a queued submission."""

    DISPATCHING = "dispatching"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"


class MetricDirection(StrEnum):
    """How a task's primary metric should be ranked."""

    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


class SubJudgeBackend(StrEnum):
    DOCKER = "docker"
    SLURM = "slurm"


class Submission(BaseModel):
    """An immutable repository revision submitted for one task."""

    model_config = ConfigDict(extra="forbid")

    repo_url: str = Field(min_length=1)
    commit_sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    task_id: str = Field(min_length=1)
    github_actor: str = Field(min_length=1)

    @field_validator("repo_url")
    @classmethod
    def validate_repo_url(cls, value: str) -> str:
        pattern = r"https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:\.git)?"
        if re.fullmatch(pattern, value) is None:
            raise ValueError("repo_url must be an HTTPS GitHub repository URL")
        return value


class TestResult(BaseModel):
    """The outcome of one named correctness check."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    passed: bool
    message: str | None = None


class JudgeResult(BaseModel):
    """A common result shape for pass/fail and scored tasks."""

    model_config = ConfigDict(extra="forbid")

    passed: bool | None = None
    score: float | None = Field(default=None, allow_inf_nan=False)
    metrics: dict[str, float] = Field(default_factory=dict)
    tests: list[TestResult] = Field(default_factory=list)


class Resources(BaseModel):
    """Resource limits requested by a task."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    cpus: int = Field(default=2, ge=1)
    memory_gb: int = Field(default=8, ge=1)
    gpus: int = Field(default=0, ge=0)
    timeout_seconds: int = Field(default=300, ge=1)


class SubJudge(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z][a-z0-9-]{0,62}$")
    backend: SubJudgeBackend
    task_ids: list[str] = Field(min_length=1)
    max_gpus: int = Field(ge=0)
    judge_revision: str = Field(min_length=1)
    registered_at: datetime


class Job(BaseModel):
    """A submission and its persistent judge state."""

    model_config = ConfigDict(extra="forbid")

    id: str
    submission: Submission
    status: JobStatus
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    result: JudgeResult | None = None
    error: str | None = None
    wandb_run_id: str | None = None
    wandb_url: str | None = None
    assigned_judge_id: str | None = None
    slurm_job_id: str | None = None
