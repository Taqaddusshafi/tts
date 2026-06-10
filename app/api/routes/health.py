"""Health endpoint."""

from fastapi import APIRouter, Request

router = APIRouter(tags=["health"])


@router.get("/health")
async def health(request: Request) -> dict[str, bool | str]:
    """Return basic service and model readiness information."""

    registry = request.app.state.model_registry
    return {
        "status": "healthy",
        "model_loaded": registry.any_loaded(),
    }

