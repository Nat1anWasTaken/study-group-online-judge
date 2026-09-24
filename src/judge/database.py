import json
import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from fcntl import LOCK_EX, LOCK_UN, flock
from importlib.resources import files
from pathlib import Path
from uuid import uuid4

from judge.models import (
    Job,
    JobStatus,
    JudgeResult,
    SubJudge,
    SubJudgeBackend,
    Submission,
)

MIGRATIONS = ("001_initial.sql", "002_sub_judges.sql")


def migrate_database(path: Path) -> None:
    """Apply each pending database migration in order."""

    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f"{path.name}.migrate.lock")
    with lock_path.open("a") as lock:
        flock(lock, LOCK_EX)
        try:
            _migrate_database(path)
        finally:
            flock(lock, LOCK_UN)


def _migrate_database(path: Path) -> None:
    with closing(_connect(path)) as connection:
        connection.execute("PRAGMA journal_mode = WAL")
        current_version = connection.execute("PRAGMA user_version").fetchone()[0]
        if current_version > len(MIGRATIONS):
            raise RuntimeError(
                f"Database schema version {current_version} is newer than this judge"
            )

        for version, filename in enumerate(MIGRATIONS, start=1):
            if version <= current_version:
                continue

            migration = (
                files("judge.migrations").joinpath(filename).read_text(encoding="utf-8")
            )
            try:
                connection.executescript(
                    f"BEGIN IMMEDIATE;\n{migration}\nPRAGMA user_version = {version};\nCOMMIT;"
                )
            except Exception:
                if connection.in_transaction:
                    connection.rollback()
                raise


def create_job(path: Path, submission: Submission) -> Job:
    """Persist a queued submission and return its job record."""

    job = Job(
        id=uuid4().hex,
        submission=submission,
        status=JobStatus.QUEUED,
        created_at=datetime.now(UTC),
    )

    with closing(_connect(path)) as connection, connection:
        connection.execute(
            """
            INSERT INTO jobs (
                id,
                repo_url,
                commit_sha,
                task_id,
                github_actor,
                status,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job.id,
                submission.repo_url,
                submission.commit_sha,
                submission.task_id,
                submission.github_actor,
                job.status.value,
                job.created_at.isoformat(),
            ),
        )

    return job


def register_sub_judge(path: Path, sub_judge: SubJudge) -> SubJudge:
    """Save a sub-judge's declared capabilities on registration."""

    with closing(_connect(path)) as connection, connection:
        connection.execute(
            """
            INSERT INTO sub_judges (
                id, backend, task_ids_json, max_gpus, judge_revision, registered_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                backend = excluded.backend,
                task_ids_json = excluded.task_ids_json,
                max_gpus = excluded.max_gpus,
                judge_revision = excluded.judge_revision,
                registered_at = excluded.registered_at
            """,
            (
                sub_judge.id,
                sub_judge.backend.value,
                json.dumps(sub_judge.task_ids),
                sub_judge.max_gpus,
                sub_judge.judge_revision,
                sub_judge.registered_at.isoformat(),
            ),
        )
    return sub_judge


def get_sub_judge(path: Path, judge_id: str) -> SubJudge | None:
    with closing(_connect(path)) as connection:
        row = connection.execute(
            "SELECT * FROM sub_judges WHERE id = ?", (judge_id,)
        ).fetchone()
    return None if row is None else _sub_judge_from_row(row)


def list_sub_judges(path: Path) -> list[SubJudge]:
    with closing(_connect(path)) as connection:
        rows = connection.execute("SELECT * FROM sub_judges ORDER BY id").fetchall()
    return [_sub_judge_from_row(row) for row in rows]


def create_remote_job(
    path: Path,
    submission: Submission,
    *,
    judge_id: str,
    request_key: str,
) -> Job:
    """Reserve a remote job, reusing its ID when the same request is retried."""

    if not request_key:
        raise ValueError("request_key must not be empty")

    with closing(_connect(path)) as connection, connection:
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute(
            "SELECT * FROM jobs WHERE repo_url = ? AND request_key = ?",
            (submission.repo_url, request_key),
        ).fetchone()
        if row is not None:
            existing = _job_from_row(row)
            if existing.submission != submission:
                raise ValueError("request_key was already used for another submission")
            return existing

        registered = connection.execute(
            "SELECT 1 FROM sub_judges WHERE id = ?", (judge_id,)
        ).fetchone()
        if registered is None:
            raise ValueError(f"Unknown sub-judge: {judge_id}")

        job = Job(
            id=uuid4().hex,
            submission=submission,
            status=JobStatus.DISPATCHING,
            created_at=datetime.now(UTC),
            assigned_judge_id=judge_id,
        )
        connection.execute(
            """
            INSERT INTO jobs (
                id, repo_url, commit_sha, task_id, github_actor,
                status, created_at, assigned_judge_id, request_key
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job.id,
                submission.repo_url,
                submission.commit_sha,
                submission.task_id,
                submission.github_actor,
                job.status.value,
                job.created_at.isoformat(),
                judge_id,
                request_key,
            ),
        )
    return job


def mark_remote_job_queued(
    path: Path, job_id: str, slurm_job_id: str | None = None
) -> Job:
    """Record a sub-judge's acceptance of an assigned job."""

    if slurm_job_id == "":
        raise ValueError("slurm_job_id must not be empty")

    with closing(_connect(path)) as connection, connection:
        cursor = connection.execute(
            """
            UPDATE jobs SET status = ?, slurm_job_id = ?
            WHERE id = ? AND status = ? AND assigned_judge_id IS NOT NULL
            """,
            (
                JobStatus.QUEUED.value,
                slurm_job_id,
                job_id,
                JobStatus.DISPATCHING.value,
            ),
        )
        if cursor.rowcount != 1:
            row = connection.execute(
                "SELECT * FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
            if (
                row is None
                or row["assigned_judge_id"] is None
                or row["slurm_job_id"] != slurm_job_id
                or row["status"]
                not in {
                    JobStatus.QUEUED.value,
                    JobStatus.RUNNING.value,
                    JobStatus.COMPLETED.value,
                    JobStatus.ERROR.value,
                }
                or (
                    row["status"] == JobStatus.ERROR.value
                    and row["slurm_job_id"] is None
                )
            ):
                raise RuntimeError(f"Job {job_id!r} cannot be marked queued")

    job = get_job(path, job_id)
    if job is None:
        raise RuntimeError(f"Job {job_id!r} disappeared from the database")
    return job


def get_job(path: Path, job_id: str) -> Job | None:
    """Load a job by ID, or return ``None`` when it does not exist."""

    with closing(_connect(path)) as connection:
        row = connection.execute(
            "SELECT * FROM jobs WHERE id = ?",
            (job_id,),
        ).fetchone()

    return None if row is None else _job_from_row(row)


def claim_next_job(path: Path) -> Job | None:
    """Atomically move the oldest queued job into the running state."""

    with closing(_connect(path)) as connection:
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT * FROM jobs
                WHERE status = ? AND assigned_judge_id IS NULL
                ORDER BY created_at, id
                LIMIT 1
                """,
                (JobStatus.QUEUED.value,),
            ).fetchone()
            if row is None:
                connection.commit()
                return None

            started_at = datetime.now(UTC).isoformat()
            connection.execute(
                """
                UPDATE jobs
                SET status = ?, started_at = ?
                WHERE id = ? AND status = ? AND assigned_judge_id IS NULL
                """,
                (
                    JobStatus.RUNNING.value,
                    started_at,
                    row["id"],
                    JobStatus.QUEUED.value,
                ),
            )
            claimed = connection.execute(
                "SELECT * FROM jobs WHERE id = ?",
                (row["id"],),
            ).fetchone()
            connection.commit()
        except Exception:
            connection.rollback()
            raise

    if claimed is None:
        raise RuntimeError("Claimed job disappeared from the database")
    return _job_from_row(claimed)


def set_wandb_run(
    path: Path,
    job_id: str,
    *,
    run_id: str,
    url: str | None,
) -> Job:
    """Attach a W&B run to a running job."""

    with closing(_connect(path)) as connection, connection:
        cursor = connection.execute(
            """
            UPDATE jobs
            SET wandb_run_id = ?, wandb_url = ?
            WHERE id = ? AND status = ?
            """,
            (run_id, url, job_id, JobStatus.RUNNING.value),
        )
        if cursor.rowcount != 1:
            raise RuntimeError(f"Job {job_id!r} is not running")

    job = get_job(path, job_id)
    if job is None:
        raise RuntimeError(f"Job {job_id!r} disappeared from the database")
    return job


def complete_job(path: Path, job_id: str, result: JudgeResult) -> Job:
    """Store a valid result and mark a running job completed."""

    return _finish_job(
        path,
        job_id,
        status=JobStatus.COMPLETED,
        result_json=result.model_dump_json(),
        error=None,
    )


def fail_job(path: Path, job_id: str, error: str) -> Job:
    """Store an infrastructure error and mark a running job failed."""

    return _finish_job(
        path,
        job_id,
        status=JobStatus.ERROR,
        result_json=None,
        error=error,
    )


def fail_remote_dispatch(path: Path, job_id: str, error: str) -> Job:
    """Record a definite failure before Slurm accepted a remote job."""

    finished_at = datetime.now(UTC).isoformat()
    with closing(_connect(path)) as connection, connection:
        cursor = connection.execute(
            """
            UPDATE jobs SET status = ?, finished_at = ?, error = ?
            WHERE id = ? AND status = ? AND assigned_judge_id IS NOT NULL
            """,
            (
                JobStatus.ERROR.value,
                finished_at,
                error,
                job_id,
                JobStatus.DISPATCHING.value,
            ),
        )
        if cursor.rowcount != 1:
            raise RuntimeError(f"Job {job_id!r} is not dispatching")

    job = get_job(path, job_id)
    if job is None:
        raise RuntimeError(f"Job {job_id!r} disappeared from the database")
    return job


def _finish_job(
    path: Path,
    job_id: str,
    *,
    status: JobStatus,
    result_json: str | None,
    error: str | None,
) -> Job:
    finished_at = datetime.now(UTC).isoformat()
    with closing(_connect(path)) as connection, connection:
        cursor = connection.execute(
            """
            UPDATE jobs
            SET status = ?, finished_at = ?, result_json = ?, error = ?
            WHERE id = ? AND status = ?
            """,
            (
                status.value,
                finished_at,
                result_json,
                error,
                job_id,
                JobStatus.RUNNING.value,
            ),
        )
        if cursor.rowcount != 1:
            raise RuntimeError(f"Job {job_id!r} is not running")

    job = get_job(path, job_id)
    if job is None:
        raise RuntimeError(f"Job {job_id!r} disappeared from the database")
    return job


def _connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA busy_timeout = 30000")
    return connection


def _job_from_row(row: sqlite3.Row) -> Job:
    result = None
    if row["result_json"] is not None:
        result = JudgeResult.model_validate(json.loads(row["result_json"]))

    return Job(
        id=row["id"],
        submission=Submission(
            repo_url=row["repo_url"],
            commit_sha=row["commit_sha"],
            task_id=row["task_id"],
            github_actor=row["github_actor"],
        ),
        status=JobStatus(row["status"]),
        created_at=row["created_at"],
        started_at=row["started_at"],
        finished_at=row["finished_at"],
        result=result,
        error=row["error"],
        wandb_run_id=row["wandb_run_id"],
        wandb_url=row["wandb_url"],
        assigned_judge_id=row["assigned_judge_id"],
        slurm_job_id=row["slurm_job_id"],
    )


def _sub_judge_from_row(row: sqlite3.Row) -> SubJudge:
    return SubJudge(
        id=row["id"],
        backend=SubJudgeBackend(row["backend"]),
        task_ids=json.loads(row["task_ids_json"]),
        max_gpus=row["max_gpus"],
        judge_revision=row["judge_revision"],
        registered_at=row["registered_at"],
    )
