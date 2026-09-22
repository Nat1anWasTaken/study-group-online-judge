import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from judge.database import create_job, get_job, migrate_database
from judge.models import JobStatus, Submission


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


if __name__ == "__main__":
    unittest.main()
