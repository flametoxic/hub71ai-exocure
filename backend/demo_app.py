"""CURE — Hub71+ demo: uvicorn demo_app:app. The needed part of the CURE core is vendored in api_gateway/; no external repo.

Windows: /app.html — phone · /hub.html — home hub · /console.html — stage console and city
"""
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from arrival.api import router
from arrival.config import Settings
from addon.routes import router as context_router


def create_app(*, settings: Settings | None = None, base_dir: Path | None = None) -> FastAPI:
    cfg = settings or Settings.from_env()
    root_dir = Path(base_dir) if base_dir is not None else Path(__file__).parent
    web_dir = root_dir / "web"
    web_mounted = cfg.serve_web and web_dir.is_dir()

    application = FastAPI(title="CURE — personal intelligence (Hub71+ demo)")
    application.add_middleware(
        CORSMiddleware,
        allow_origins=list(cfg.cors_origins),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    application.include_router(router)
    application.include_router(context_router)

    @application.get("/health")
    def health():
        ready, reason = cfg.language_ready()
        payload = {"ready": ready, "language_mode": cfg.language_mode, "web_mounted": web_mounted}
        if cfg.language_mode == "openai":
            payload["model"] = cfg.openai_model
        if reason:
            payload["reason"] = reason
        return JSONResponse(payload, status_code=200 if ready else 503)

    @application.get("/")
    def root():
        if web_mounted:
            return RedirectResponse("/console.html")
        return {"service": "CURE backend", "health": "/health", "api": "/twin"}

    if web_mounted:
        application.mount("/", StaticFiles(directory=web_dir, html=True), name="web")
    return application


app = create_app()
