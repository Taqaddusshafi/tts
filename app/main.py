"""FastAPI application entrypoint."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from app.api.routes.health import router as health_router
from app.api.routes.tts import router as tts_router
from app.core.config import get_settings
from app.core.errors import AppError, ErrorEnvelope
from app.core.logger import configure_logging, get_logger
from app.models.loaders import build_model_registry
from app.services.tts_service import TTSService

settings = get_settings()
configure_logging(settings.log_level)
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialize and clean up app-scoped model resources."""

    logger.info("service_startup_begin", app=settings.app_name, version=settings.app_version)
    registry = build_model_registry(settings)
    app.state.model_registry = registry
    app.state.tts_service = TTSService(registry=registry, settings=settings)

    if settings.load_models_on_startup:
        registry.preload()
        if settings.warmup_on_startup:
            registry.warmup()

    logger.info("service_startup_complete")
    try:
        yield
    finally:
        logger.info("service_shutdown_begin")
        logger.info("service_shutdown_complete")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan,
)


@app.exception_handler(AppError)
async def app_error_handler(_: Request, exc: AppError) -> JSONResponse:
    """Render application errors through the standard envelope."""

    logger.warning(
        "app_error",
        code=exc.code,
        message=exc.message,
        status_code=exc.status_code,
        details=exc.details,
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorEnvelope(exc.code, exc.message, exc.details).to_dict(),
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    """Render FastAPI/Pydantic validation errors through the standard envelope."""

    return JSONResponse(
        status_code=422,
        content=jsonable_encoder(ErrorEnvelope(
            "validation_error",
            "Request validation failed.",
            {"errors": exc.errors()},
        ).to_dict()),
    )


@app.exception_handler(HTTPException)
async def http_error_handler(_: Request, exc: HTTPException) -> JSONResponse:
    """Render HTTP exceptions through the standard envelope."""

    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorEnvelope(
            "http_error",
            str(exc.detail),
        ).to_dict(),
    )


@app.exception_handler(Exception)
async def unhandled_error_handler(_: Request, exc: Exception) -> JSONResponse:
    """Render unhandled failures without leaking internals to clients."""

    logger.exception("unhandled_error", error=str(exc))
    return JSONResponse(
        status_code=500,
        content=ErrorEnvelope(
            "internal_error",
            "An unexpected error occurred.",
        ).to_dict(),
    )


app.include_router(health_router)
app.include_router(tts_router)
