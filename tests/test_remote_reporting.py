import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from judge.models import JudgeResult, Resources, Submission
from judge.remote_reporter import publish_report
from judge.ssh import RemoteConfig, RemoteRequest, RemoteSnapshot


class RemoteReportingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / ".netrc").write_text(
            "machine api.wandb.ai login judge password remote-test-key\n"
        )
        self.request = RemoteRequest(
            job_id="1" * 32,
            submission=Submission(
                repo_url="https://github.com/example/repo.git",
                commit_sha="a" * 40,
                task_id="lab2",
                github_actor="student",
            ),
            config=RemoteConfig(host="nano4", user="judge", work_root=str(self.root)),
            resources=Resources(gpus=1),
        )
        self.workspace = self.root / "jobs" / self.request.job_id
        self.output = self.workspace / "output"
        self.output.mkdir(parents=True)
        self.record: dict = {"state": "preparing"}
        self.wandb_run = Mock(url="https://wandb.ai/entity/project/runs/test")
        self.wandb_run.summary = {}
        init = patch("judge.remote_reporter.wandb.init", return_value=self.wandb_run)
        self.init = init.start()
        self.addCleanup(init.stop)

    def publish(self, snapshot=None):
        snapshot = snapshot or RemoteSnapshot()
        publish_report(self.request, snapshot, self.workspace, self.record)
        return snapshot

    def test_uses_remote_netrc_and_reports_setup_failure_without_slurm(self):
        snapshot = self.publish(RemoteSnapshot(error="uv sync failed"))
        self.assertEqual(self.wandb_run.summary["error"], "uv sync failed")
        self.assertEqual(self.wandb_run.summary["judge_status"], "error")
        self.assertTrue(self.record["report"]["complete"])
        self.assertEqual(snapshot.wandb_url, self.wandb_run.url)
        settings = self.init.call_args.kwargs["settings"]
        self.assertEqual(settings.api_key, "remote-test-key")
        self.assertEqual(settings.finish_timeout, 30)
        self.assertTrue(settings.finish_timeout_raises)
        self.assertEqual(self.init.call_args.kwargs["id"], self.request.job_id)
        self.assertNotIn("api_key", self.init.call_args.kwargs["config"])

    def test_missing_or_wrong_credentials_never_prompt(self):
        (self.root / ".netrc").unlink()
        with self.assertRaises(FileNotFoundError):
            self.publish()
        (self.root / ".netrc").write_text("machine unrelated.example password wrong")
        with self.assertRaisesRegex(ValueError, "api.wandb.ai"):
            self.publish()
        self.init.assert_not_called()

    def test_report_offsets_advance_only_after_upload_and_skip_duplicate_polls(self):
        text = "模型完成\n"
        (self.output / "setup.log").write_text(text)
        self.wandb_run.finish.side_effect = RuntimeError("network disconnected")
        with self.assertRaisesRegex(RuntimeError, "disconnected"):
            self.publish()
        self.assertNotIn("report", self.record)
        self.wandb_run.finish.side_effect = None
        with patch("builtins.print") as output:
            self.publish()
        self.assertEqual(self.record["report"]["setup_offset"], len(text.encode()))
        output.assert_any_call(text, end="", flush=True)
        self.init.reset_mock()
        self.publish()
        self.init.assert_not_called()

    def test_large_remote_log_drains_before_final_result_and_does_not_return_text(self):
        (self.output / "slurm.log").write_text("x" * 70000)
        (self.output / "result.json").write_text('{"passed":true}')
        self.record["state"] = "submitted"
        snapshot = RemoteSnapshot(slurm_job_id="123", result=JudgeResult(passed=True))
        with (
            patch("judge.remote_reporter._publish_result") as publish,
            patch("builtins.print"),
        ):
            first = self.publish(snapshot)
            self.assertTrue(first.report_pending)
            self.assertFalse(self.record["report"].get("complete"))
            publish.assert_not_called()
            second = self.publish(
                RemoteSnapshot(slurm_job_id="123", result=snapshot.result)
            )
            self.assertFalse(second.report_pending)
            publish.assert_called_once()
        self.assertEqual(first.slurm_log, "")
        self.assertTrue(self.record["report"]["complete"])
        self.init.reset_mock()
        self.publish(RemoteSnapshot(result=snapshot.result))
        self.init.assert_not_called()
