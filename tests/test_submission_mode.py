import sys
from threading import Event
from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient

from judge import ssh_worker, worker
from judge.database import create_job, get_job, migrate_database
from judge.main import create_app
from judge.models import Submission


@pytest.mark.parametrize("value", [None, "false", "", "typo"])
def test_leaderboard_only_mode(tmp_path, monkeypatch, value):
    monkeypatch.delenv("JUDGE_ALLOW_CODE_SUBMISSIONS", raising=False)
    if value is not None:
        monkeypatch.setenv("JUDGE_ALLOW_CODE_SUBMISSIONS", value)
    monkeypatch.delenv("JUDGE_API_TOKEN", raising=False)
    # Disabled submission configuration must not prevent leaderboard startup.
    monkeypatch.setenv("JUDGE_SSH_HOST", "invalid host")
    database_path = tmp_path / "judge.db"
    monkeypatch.setenv("JUDGE_DATABASE_PATH", str(database_path))
    monkeypatch.setenv("JUDGE_UI_DIST", str(tmp_path / "no-ui"))
    monkeypatch.delenv("JUDGE_SLACK_WEBHOOK_URL", raising=False)
    monkeypatch.delenv("JUDGE_SLACK_BOT_TOKEN", raising=False)
    refreshed = Event()

    from judge.leaderboard import LeaderboardService

    refresh = LeaderboardService.refresh

    def refresh_and_signal(service):
        refresh(service)
        refreshed.set()

    with (
        patch("judge.main.load_dotenv"),
        patch("judge.leaderboard.flock_submissions", return_value=[]) as fetch,
        patch.object(LeaderboardService, "refresh", refresh_and_signal),
        TestClient(create_app()) as client,
    ):
        assert refreshed.wait(timeout=5)
        fetch.assert_called_once()
        response = client.get("/api/leaderboard")
        assert response.status_code == 200
        assert response.json()["source"] == "flock"
        assert response.json()["updated_at"] is not None
        assert response.json()["error"] is None
        assert client.get("/healthz").status_code == 200
        assert set(client.get("/openapi.json").json()["paths"]) == {
            "/healthz",
            "/api/leaderboard",
        }
        for headers in ({}, {"Authorization": "Bearer secret"}):
            response = client.post("/submissions", headers=headers, json={})
            assert response.status_code == 404
            assert client.get("/jobs/existing", headers=headers).status_code == 404
    assert not database_path.exists()


def test_router_registration_is_fixed_per_application(tmp_path, monkeypatch):
    monkeypatch.setenv("JUDGE_UI_DIST", str(tmp_path))
    with patch("judge.main.load_dotenv"):
        monkeypatch.setenv("JUDGE_ALLOW_CODE_SUBMISSIONS", "true")
        enabled_app = create_app()
        monkeypatch.setenv("JUDGE_ALLOW_CODE_SUBMISSIONS", "false")
        disabled_app = create_app()

    assert enabled_app.state.allow_code_submissions is True
    assert disabled_app.state.allow_code_submissions is False
    assert "/submissions" in enabled_app.openapi()["paths"]
    assert "/jobs/{job_id}" in enabled_app.openapi()["paths"]
    assert set(disabled_app.openapi()["paths"]) == {"/healthz", "/api/leaderboard"}


def test_disabled_workers_leave_queued_jobs_untouched(tmp_path, monkeypatch):
    monkeypatch.delenv("JUDGE_ALLOW_CODE_SUBMISSIONS", raising=False)
    database_path = tmp_path / "judge.db"
    migrate_database(database_path)
    job = create_job(
        database_path,
        Submission(
            repo_url="https://github.com/example/repo.git",
            commit_sha="a" * 40,
            task_id="lab1",
            github_actor="student",
        ),
    )
    with patch("judge.worker.DockerExecutor") as executor:
        assert (
            worker.run_worker(
                database_path=database_path,
                work_root=tmp_path / "work",
                runner_image="judge:test",
                wandb_project="test",
                once=True,
            )
            is None
        )
        executor.assert_not_called()
    assert get_job(database_path, job.id) == job
    transport = Mock()
    with patch("judge.ssh_worker.pending_ssh_jobs") as pending:
        ssh_worker.poll_jobs(database_path, transport)
        pending.assert_not_called()
        transport.poll.assert_not_called()


@pytest.mark.parametrize("module", [worker, ssh_worker])
def test_disabled_worker_entrypoints_skip_initialization(tmp_path, monkeypatch, module):
    monkeypatch.setenv("JUDGE_ALLOW_CODE_SUBMISSIONS", "false")
    monkeypatch.setenv("JUDGE_DATABASE_PATH", str(tmp_path / "judge.db"))
    monkeypatch.delenv("JUDGE_RUNNER_IMAGE", raising=False)
    monkeypatch.delenv("WANDB_PROJECT", raising=False)
    monkeypatch.setattr(sys, "argv", ["worker", "--once"])
    with patch.object(module, "migrate_database") as migrate:
        module.main()
        migrate.assert_not_called()


def test_enabled_cpu_worker_polls_jobs(tmp_path, monkeypatch):
    monkeypatch.setenv("JUDGE_ALLOW_CODE_SUBMISSIONS", "true")
    with (
        patch("judge.worker.migrate_database"),
        patch("judge.worker.DockerExecutor"),
        patch("judge.worker.claim_next_job", return_value=None) as claim,
    ):
        worker.run_worker(
            database_path=tmp_path / "judge.db",
            work_root=tmp_path / "work",
            runner_image="judge:test",
            wandb_project="test",
            once=True,
        )
        claim.assert_called_once()
