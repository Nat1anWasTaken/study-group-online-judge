import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import Mock, patch

from judge.database import (
    append_remote_event,
    create_remote_job,
    get_job,
    mark_remote_event_reported,
    migrate_database,
    next_unreported_event,
    register_sub_judge,
)
from judge.models import (
    JobStatus,
    JudgeResult,
    RemoteEvent,
    RemoteEventKind,
    SubJudge,
    SubJudgeBackend,
    Submission,
)
from judge.remote_reporter import run_reporter


class RemoteReportingTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "judge.db"
        migrate_database(self.path)
        register_sub_judge(
            self.path,
            SubJudge(
                id="nano4",
                backend=SubJudgeBackend.SLURM,
                task_ids=["lab1"],
                max_gpus=8,
                judge_revision="test",
                registered_at=datetime.now(UTC),
            ),
        )
        self.job = create_remote_job(
            self.path,
            Submission(
                repo_url="https://github.com/example/repo.git",
                commit_sha="a" * 40,
                task_id="lab1",
                github_actor="participant",
            ),
            judge_id="nano4",
            request_key="request-1",
        )

    def event(
        self,
        sequence: int,
        kind: RemoteEventKind,
        *,
        line: str | None = None,
        result: JudgeResult | None = None,
        error: str | None = None,
    ) -> RemoteEvent:
        return RemoteEvent(
            sequence=sequence,
            kind=kind,
            slurm_job_id="12345",
            line=line,
            result=result,
            error=error,
        )

    def test_ordered_events_are_idempotent_and_change_job_state(self) -> None:
        started = self.event(1, RemoteEventKind.STARTED)
        running = append_remote_event(self.path, "nano4", self.job.id, started)
        self.assertEqual(running.status, JobStatus.RUNNING)
        self.assertEqual(running.slurm_job_id, "12345")
        self.assertEqual(
            append_remote_event(self.path, "nano4", self.job.id, started), running
        )
        with self.assertRaisesRegex(ValueError, "Expected event sequence 2"):
            append_remote_event(
                self.path,
                "nano4",
                self.job.id,
                self.event(3, RemoteEventKind.LOG, line="skipped"),
            )
        with self.assertRaisesRegex(ValueError, "different data"):
            append_remote_event(
                self.path,
                "nano4",
                self.job.id,
                self.event(1, RemoteEventKind.LOG, line="changed"),
            )
        append_remote_event(
            self.path,
            "nano4",
            self.job.id,
            self.event(2, RemoteEventKind.LOG, line="training\n"),
        )
        finished = append_remote_event(
            self.path,
            "nano4",
            self.job.id,
            self.event(3, RemoteEventKind.COMPLETED, result=JudgeResult(passed=True)),
        )
        self.assertEqual(finished.status, JobStatus.COMPLETED)
        self.assertTrue(finished.result and finished.result.passed)
        with self.assertRaisesRegex(ValueError, "not running"):
            append_remote_event(
                self.path,
                "nano4",
                self.job.id,
                self.event(4, RemoteEventKind.LOG, line="late"),
            )

    def test_report_cursor_delivers_each_event_in_order(self) -> None:
        append_remote_event(
            self.path, "nano4", self.job.id, self.event(1, RemoteEventKind.STARTED)
        )
        append_remote_event(
            self.path,
            "nano4",
            self.job.id,
            self.event(2, RemoteEventKind.FAILED, error="ValueError: bad model"),
        )
        first = next_unreported_event(self.path)
        assert first is not None
        self.assertEqual(first[1].sequence, 1)
        mark_remote_event_reported(self.path, self.job.id, 1)
        second = next_unreported_event(self.path)
        assert second is not None
        self.assertEqual(second[1].sequence, 2)
        mark_remote_event_reported(self.path, self.job.id, 2)
        self.assertIsNone(next_unreported_event(self.path))
        job = get_job(self.path, self.job.id)
        assert job is not None
        self.assertEqual(job.error, "ValueError: bad model")

    def test_reporter_marks_event_only_after_wandb_succeeds(self) -> None:
        append_remote_event(
            self.path, "nano4", self.job.id, self.event(1, RemoteEventKind.STARTED)
        )
        with (
            patch(
                "judge.remote_reporter.publish_event",
                side_effect=RuntimeError("offline"),
            ),
            self.assertRaisesRegex(RuntimeError, "offline"),
        ):
            run_reporter(
                database_path=self.path, wandb_project="study-group", once=True
            )
        self.assertIsNotNone(next_unreported_event(self.path))
        with patch("judge.remote_reporter.publish_event", Mock()) as publish:
            run_reporter(
                database_path=self.path, wandb_project="study-group", once=True
            )
        publish.assert_called_once()
        self.assertIsNone(next_unreported_event(self.path))


if __name__ == "__main__":
    unittest.main()
