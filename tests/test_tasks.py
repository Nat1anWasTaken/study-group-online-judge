import unittest
from pathlib import Path

from judge.models import JudgeResult
from judge.tasks import TASKS
from judge.tasks.base import Task


class ExampleTask(Task):
    id = "example"

    def evaluate(self, submission: Path) -> JudgeResult:
        return JudgeResult(passed=submission.is_dir())


class TaskRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original_tasks = TASKS.copy()
        TASKS.clear()

    def tearDown(self) -> None:
        TASKS.clear()
        TASKS.update(self.original_tasks)

    def test_tasks_are_registered_in_a_plain_dictionary(self) -> None:
        task = ExampleTask()
        TASKS[task.id] = task

        self.assertIs(TASKS["example"], task)


if __name__ == "__main__":
    unittest.main()
