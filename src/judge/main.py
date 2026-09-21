from fastapi import FastAPI

from judge.routers import health

app = FastAPI()

app.include_router(health.router)
