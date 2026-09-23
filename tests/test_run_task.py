import tempfile
import unittest
from pathlib import Path

from judge.models import JudgeResult
from judge.run_task import run_task
from judge.tasks import TASKS
from judge.tasks.base import Task


class ExampleTask(Task):
    id = "example"

    def evaluate(self, submission: Path) -> JudgeResult:
        return JudgeResult(passed=(submission / "solution.py").is_file())


class TaskRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.submission = self.root / "submission"
        self.submission.mkdir()
        (self.submission / "solution.py").touch()

        self.original_tasks = TASKS.copy()
        TASKS.clear()
        TASKS[ExampleTask.id] = ExampleTask()
        self.addCleanup(self.restore_tasks)

    def restore_tasks(self) -> None:
        TASKS.clear()
        TASKS.update(self.original_tasks)

    def test_writes_a_machine_readable_result(self) -> None:
        output = self.root / "result.json"

        result = run_task("example", self.submission, output)

        persisted = JudgeResult.model_validate_json(output.read_text())
        self.assertEqual(persisted, result)
        self.assertTrue(persisted.passed)

    def test_rejects_an_unknown_task_without_writing_a_result(self) -> None:
        output = self.root / "result.json"

        with self.assertRaisesRegex(ValueError, "Unknown task"):
            run_task("missing", self.submission, output)

        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
