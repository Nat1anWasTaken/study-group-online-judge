import asyncio
import logging
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager, suppress
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from judge.config import allow_code_submissions
from judge.database import migrate_database
from judge.execution_logging import configure_logging
from judge.leaderboard import LeaderboardService
from judge.routers import health, leaderboard, submissions
from judge.ssh import RemoteConfig


@asynccontextmanager
async def lifespan(judge_app: FastAPI) -> AsyncGenerator[None]:
    configure_logging()

    api_token = os.environ.get("JUDGE_API_TOKEN")
    if judge_app.state.allow_code_submissions and not api_token:
        raise RuntimeError("JUDGE_API_TOKEN must be configured")

    database_path = Path(os.environ.get("JUDGE_DATABASE_PATH", "data/judge.db"))
    if judge_app.state.allow_code_submissions:
        migrate_database(database_path)
    judge_app.state.api_token = api_token
    judge_app.state.remote_config = (
        RemoteConfig.from_environment()
        if judge_app.state.allow_code_submissions
        else None
    )
    judge_app.state.database_path = database_path
    judge_app.state.leaderboard = LeaderboardService(database_path)
    interval = max(10, float(os.getenv("JUDGE_LEADERBOARD_REFRESH_SECONDS", "60")))

    async def refresh_leaderboard():
        while True:
            try:
                await asyncio.to_thread(judge_app.state.leaderboard.refresh)
            except OSError:
                logging.getLogger("uvicorn.error.judge").exception(
                    "Unable to persist leaderboard snapshot"
                )
            await asyncio.sleep(interval)

    poller = asyncio.create_task(refresh_leaderboard())

    try:
        yield
    finally:
        poller.cancel()
        with suppress(asyncio.CancelledError):
            await poller


def create_app() -> FastAPI:
    load_dotenv()
    judge_app = FastAPI(lifespan=lifespan)
    judge_app.state.allow_code_submissions = allow_code_submissions()
    judge_app.include_router(health.router)
    if judge_app.state.allow_code_submissions:
        judge_app.include_router(submissions.router)
    judge_app.include_router(leaderboard.router)

    # Register API routes before the optional frontend catch-all.
    ui_dist = Path(
        os.getenv(
            "JUDGE_UI_DIST", str(Path(__file__).resolve().parents[2] / "ui" / "dist")
        )
    )
    if ui_dist.is_dir():
        judge_app.mount("/", StaticFiles(directory=ui_dist, html=True), name="ui")
    return judge_app


app = create_app()
