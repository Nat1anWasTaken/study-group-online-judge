import json
import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from fcntl import LOCK_EX, LOCK_UN, flock
from importlib.resources import files
from pathlib import Path
from uuid import uuid4

from judge.models import Job, JobStatus, JudgeResult, Submission

MIGRATIONS = ("001_initial.sql",)


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
                WHERE status = ?
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
                WHERE id = ? AND status = ?
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
    )
