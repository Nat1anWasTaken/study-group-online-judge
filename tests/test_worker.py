import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from judge.database import claim_next_job, create_job, get_job, migrate_database
from judge.executor import DockerExecutor, ExecutionResult
from judge.models import Job, JobStatus, JudgeResult, Submission
from judge.repository import CheckoutError
from judge.tasks import TASKS
from judge.tasks.base import Task
from judge.worker import run_job


class ExampleTask(Task):
    id = "example"

    def evaluate(self, submission: Path) -> JudgeResult:
        return JudgeResult(passed=submission.is_dir())


class WorkerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.database_path = self.root / "judge.db"
        self.work_root = self.root / "work"
        migrate_database(self.database_path)

        self.original_tasks = TASKS.copy()
        TASKS.clear()
        TASKS[ExampleTask.id] = ExampleTask()
        self.addCleanup(self.restore_tasks)

        created = create_job(
            self.database_path,
            Submission(
                repo_url="https://github.com/cerulean-works/example.git",
                commit_sha="a" * 40,
                task_id="example",
                github_actor="student",
            ),
        )
        claimed = claim_next_job(self.database_path)
        assert claimed is not None
        self.job: Job = claimed
        self.assertEqual(self.job.id, created.id)

        self.executor = Mock(spec=DockerExecutor)
        self.output: list[str] = []

    def restore_tasks(self) -> None:
        TASKS.clear()
        TASKS.update(self.original_tasks)

    @staticmethod
    def fake_checkout(
        repo_url: str,
        commit_sha: str,
        destination: Path,
    ) -> Path:
        destination.mkdir()
        return destination

    def test_persists_a_valid_result(self) -> None:
        def execute(**arguments: object) -> ExecutionResult:
            output_directory = arguments["output_directory"]
            assert isinstance(output_directory, Path)
            output_directory.mkdir()
            (output_directory / "result.json").write_text(
                JudgeResult(passed=True).model_dump_json()
            )
            return ExecutionResult(returncode=0)

        self.executor.run.side_effect = execute

        with patch("judge.worker.checkout_repository", side_effect=self.fake_checkout):
            completed = run_job(
                self.job,
                database_path=self.database_path,
                work_root=self.work_root,
                executor=self.executor,
                on_output=self.output.append,
            )

        self.assertEqual(completed.status, JobStatus.COMPLETED)
        self.assertTrue(completed.result and completed.result.passed)
        self.assertIn(f"[judge] completed job {self.job.id}\n", self.output)

    def test_persists_checkout_failures(self) -> None:
        with patch(
            "judge.worker.checkout_repository",
            side_effect=CheckoutError("clone failed"),
        ):
            failed = run_job(
                self.job,
                database_path=self.database_path,
                work_root=self.work_root,
                executor=self.executor,
                on_output=self.output.append,
            )

        persisted = get_job(self.database_path, self.job.id)
        self.assertEqual(failed.status, JobStatus.ERROR)
        self.assertEqual(persisted, failed)
        self.assertEqual(failed.error, "CheckoutError: clone failed")

    def test_persists_nonzero_runner_exits(self) -> None:
        self.executor.run.return_value = ExecutionResult(returncode=7)

        with patch("judge.worker.checkout_repository", side_effect=self.fake_checkout):
            failed = run_job(
                self.job,
                database_path=self.database_path,
                work_root=self.work_root,
                executor=self.executor,
                on_output=self.output.append,
            )

        self.assertEqual(failed.status, JobStatus.ERROR)
        self.assertEqual(failed.error, "RuntimeError: Task runner exited with status 7")


if __name__ == "__main__":
    unittest.main()
