"""TTS HTTP route."""

from fastapi import APIRouter, Request, Response
from starlette.concurrency import run_in_threadpool

from app.api.schemas.tts import TTSRequest
from app.services.tts_service import TTSService

router = APIRouter(prefix="/v1", tags=["tts"])


def get_tts_service(request: Request) -> TTSService:
    """Fetch the app-scoped TTS service."""

    return request.app.state.tts_service


@router.post("/tts", response_class=Response)
async def synthesize_tts(payload: TTSRequest, request: Request) -> Response:
    """Generate speech and return audio/wav bytes."""

    service = get_tts_service(request)
    wav_bytes = await run_in_threadpool(
        service.synthesize,
        payload.text,
        payload.language,
        payload.voice,
    )
    return Response(content=wav_bytes, media_type="audio/wav")

