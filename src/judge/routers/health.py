from fastapi import APIRouter

router = APIRouter(tags=["Healthz"])


@router.get("/healthz")
async def healthz():
    """Returns health signal for server"""

    return {"status": "ok"}
