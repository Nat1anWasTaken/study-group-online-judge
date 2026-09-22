import subprocess
import tempfile
import unittest
from pathlib import Path

from judge.repository import CheckoutError, checkout_repository, create_job_workspace


class WorkspaceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.work_root = Path(self.temporary_directory.name) / "work"

    def test_creates_a_workspace_for_a_job(self) -> None:
        job_id = "a" * 32

        workspace = create_job_workspace(self.work_root, job_id)

        self.assertEqual(workspace, self.work_root / job_id)
        self.assertTrue(workspace.is_dir())

    def test_rejects_a_job_id_that_could_escape_the_work_root(self) -> None:
        with self.assertRaises(ValueError):
            create_job_workspace(self.work_root, "../escape")

    def test_does_not_reuse_an_existing_workspace(self) -> None:
        job_id = "a" * 32
        create_job_workspace(self.work_root, job_id)

        with self.assertRaises(FileExistsError):
            create_job_workspace(self.work_root, job_id)


class CheckoutTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)

    def test_checks_out_the_submitted_commit_not_the_latest_commit(self) -> None:
        source = self.root / "source"
        source.mkdir()
        self.git("init", "--initial-branch=main", workdir=source)
        self.git("config", "user.name", "Judge Test", workdir=source)
        self.git("config", "user.email", "judge@example.com", workdir=source)

        solution = source / "solution.py"
        solution.write_text("VERSION = 1\n")
        self.git("add", "solution.py", workdir=source)
        self.git("commit", "-m", "first", workdir=source)
        submitted_commit = self.git("rev-parse", "HEAD", workdir=source).stdout.strip()

        solution.write_text("VERSION = 2\n")
        self.git("commit", "-am", "second", workdir=source)

        checkout = self.root / "checkout"
        checkout_repository(str(source), submitted_commit, checkout)

        self.assertEqual((checkout / "solution.py").read_text(), "VERSION = 1\n")
        self.assertEqual(
            self.git("rev-parse", "HEAD", workdir=checkout).stdout.strip(),
            submitted_commit,
        )

    def test_rejects_an_existing_destination(self) -> None:
        destination = self.root / "checkout"
        destination.mkdir()

        with self.assertRaises(FileExistsError):
            checkout_repository("unused", "a" * 40, destination)

    def test_removes_a_partial_checkout_after_failure(self) -> None:
        destination = self.root / "checkout"

        with self.assertRaises(CheckoutError):
            checkout_repository(
                str(self.root / "missing-repository"),
                "a" * 40,
                destination,
            )

        self.assertFalse(destination.exists())

    def git(
        self,
        *arguments: str,
        workdir: Path,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *arguments],
            cwd=workdir,
            check=True,
            capture_output=True,
            text=True,
        )


if __name__ == "__main__":
    unittest.main()
