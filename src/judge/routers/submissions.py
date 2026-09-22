from hmac import compare_digest
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from judge import database
from judge.models import Job, Submission
from judge.tasks import TASKS

bearer = HTTPBearer(auto_error=False)


def require_api_token(
    request: Request,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
) -> None:
    """Require the shared submission token configured on the judge."""

    expected_token = request.app.state.api_token
    if credentials is None or not compare_digest(
        credentials.credentials, expected_token
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API token",
            headers={"WWW-Authenticate": "Bearer"},
        )


router = APIRouter(
    tags=["Submissions"],
    dependencies=[Depends(require_api_token)],
)


@router.post("/submissions", response_model=Job, status_code=status.HTTP_201_CREATED)
def submit(submission: Submission, request: Request) -> Job:
    """Queue an immutable repository revision for judging."""

    if submission.task_id not in TASKS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Unknown task: {submission.task_id}",
        )

    return database.create_job(request.app.state.database_path, submission)


@router.get("/jobs/{job_id}", response_model=Job)
def get_job(job_id: str, request: Request) -> Job:
    """Return the current state of a submission job."""

    job = database.get_job(request.app.state.database_path, job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found",
        )
    return job
