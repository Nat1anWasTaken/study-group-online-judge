import sqlite3
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path

from judge.database import (
    claim_next_job,
    complete_job,
    create_job,
    fail_job,
    get_job,
    migrate_database,
    set_wandb_run,
)
from judge.models import JobStatus, JudgeResult, Submission


class DatabaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.database_path = Path(self.temporary_directory.name) / "judge.db"
        migrate_database(self.database_path)

    def test_applies_migrations_once(self) -> None:
        migrate_database(self.database_path)

        with closing(sqlite3.connect(self.database_path)) as connection:
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            tables = connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()

        self.assertEqual(version, 1)
        self.assertIn(("jobs",), tables)

    def test_rejects_a_database_from_a_newer_judge(self) -> None:
        with closing(sqlite3.connect(self.database_path)) as connection, connection:
            connection.execute("PRAGMA user_version = 999")

        with self.assertRaisesRegex(RuntimeError, "newer than this judge"):
            migrate_database(self.database_path)

    def test_serializes_concurrent_migration_attempts(self) -> None:
        database_path = Path(self.temporary_directory.name) / "concurrent.db"

        with ThreadPoolExecutor(max_workers=2) as executor:
            list(executor.map(lambda _: migrate_database(database_path), range(2)))

        with closing(sqlite3.connect(database_path)) as connection:
            version = connection.execute("PRAGMA user_version").fetchone()[0]

        self.assertEqual(version, 1)

    def test_round_trips_a_queued_job(self) -> None:
        submission = Submission(
            repo_url="https://github.com/cerulean-works/example.git",
            commit_sha="a" * 40,
            task_id="example",
            github_actor="student",
        )

        created = create_job(self.database_path, submission)
        loaded = get_job(self.database_path, created.id)

        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertEqual(loaded.submission, submission)
        self.assertEqual(loaded.status, JobStatus.QUEUED)

    def test_returns_none_for_an_unknown_job(self) -> None:
        self.assertIsNone(get_job(self.database_path, "missing"))

    def test_only_one_concurrent_worker_claims_a_job(self) -> None:
        create_job(self.database_path, self.submission())

        with ThreadPoolExecutor(max_workers=2) as executor:
            claimed = list(
                executor.map(lambda _: claim_next_job(self.database_path), range(2))
            )

        jobs = [job for job in claimed if job is not None]
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0].status, JobStatus.RUNNING)
        self.assertIsNotNone(jobs[0].started_at)

    def test_completes_a_running_job_with_its_result(self) -> None:
        created = create_job(self.database_path, self.submission())
        claim_next_job(self.database_path)

        completed = complete_job(
            self.database_path,
            created.id,
            JudgeResult(passed=True),
        )

        self.assertEqual(completed.status, JobStatus.COMPLETED)
        self.assertTrue(completed.result and completed.result.passed)
        self.assertIsNotNone(completed.finished_at)

    def test_attaches_a_wandb_run_to_a_running_job(self) -> None:
        created = create_job(self.database_path, self.submission())
        claim_next_job(self.database_path)

        updated = set_wandb_run(
            self.database_path,
            created.id,
            run_id="wandb-run",
            url="https://wandb.example/run",
        )

        self.assertEqual(updated.wandb_run_id, "wandb-run")
        self.assertEqual(updated.wandb_url, "https://wandb.example/run")

    def test_rejects_attaching_a_wandb_run_to_a_queued_job(self) -> None:
        created = create_job(self.database_path, self.submission())

        with self.assertRaisesRegex(RuntimeError, "is not running"):
            set_wandb_run(
                self.database_path,
                created.id,
                run_id="wandb-run",
                url=None,
            )

    def test_fails_a_running_job_with_an_error(self) -> None:
        created = create_job(self.database_path, self.submission())
        claim_next_job(self.database_path)

        failed = fail_job(self.database_path, created.id, "checkout failed")

        self.assertEqual(failed.status, JobStatus.ERROR)
        self.assertEqual(failed.error, "checkout failed")
        self.assertIsNotNone(failed.finished_at)

    @staticmethod
    def submission() -> Submission:
        return Submission(
            repo_url="https://github.com/cerulean-works/example.git",
            commit_sha="a" * 40,
            task_id="example",
            github_actor="student",
        )


if __name__ == "__main__":
    unittest.main()
