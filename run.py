"""Local development server runner."""

import uvicorn

from app.core.config import get_settings


def main() -> None:
    """Run the FastAPI app with Uvicorn."""

    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.environment == "local",
    )


if __name__ == "__main__":
    main()

