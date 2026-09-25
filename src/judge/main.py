import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from judge.agent_channel import AgentChannel
from judge.database import migrate_database
from judge.routers import agents, health, submissions


@asynccontextmanager
async def lifespan(judge_app: FastAPI) -> AsyncIterator[None]:
    api_token = os.environ.get("JUDGE_API_TOKEN")
    if not api_token:
        raise RuntimeError("JUDGE_API_TOKEN must be configured")

    database_path = Path(os.environ.get("JUDGE_DATABASE_PATH", "data/judge.db"))
    migrate_database(database_path)
    judge_app.state.api_token = api_token
    judge_app.state.agent_token = os.environ.get("JUDGE_AGENT_TOKEN") or None
    judge_app.state.judge_revision = os.environ.get("JUDGE_REVISION") or None
    judge_app.state.agent_channel = AgentChannel()
    judge_app.state.database_path = database_path
    yield


app = FastAPI(lifespan=lifespan)
app.include_router(agents.router)
app.include_router(health.router)
app.include_router(submissions.router)
