import tempfile
import unittest
from pathlib import Path

from judge.executor import DockerExecutor
from judge.models import Resources


class DockerCommandTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        root = Path(self.temporary_directory.name)
        self.submission = root / "submission"
        self.output = root / "output"
        self.submission.mkdir()
        self.output.mkdir()
        self.executor = DockerExecutor(
            "judge:test",
            user_id=1234,
            group_id=5678,
        )

    def command(self, resources: Resources) -> list[str]:
        return self.executor.build_command(
            task_id="assignment-01",
            resources=resources,
            submission=self.submission,
            output_directory=self.output,
            container_name="judge-test",
        )

    def test_applies_isolation_and_resource_limits(self) -> None:
        command = self.command(Resources(cpus=2, memory_gb=4, timeout_seconds=60))

        self.assertEqual(command[command.index("--network") + 1], "none")
        self.assertIn("--read-only", command)
        self.assertEqual(command[command.index("--cap-drop") + 1], "ALL")
        self.assertEqual(
            command[command.index("--security-opt") + 1], "no-new-privileges"
        )
        self.assertEqual(command[command.index("--cpus") + 1], "2")
        self.assertEqual(command[command.index("--memory") + 1], "4g")
        self.assertEqual(command[command.index("--user") + 1], "1234:5678")
        self.assertNotIn("--gpus", command)

    def test_requests_gpus_only_for_gpu_tasks(self) -> None:
        command = self.command(Resources(gpus=2))

        gpu_option = command.index("--gpus")
        self.assertEqual(command[gpu_option + 1], "2")

    def test_refuses_to_run_as_root(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-root"):
            DockerExecutor("judge:test", user_id=0, group_id=0)


if __name__ == "__main__":
    unittest.main()
