import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from app.database import init_db
from app.config import settings
from app.routers import (
    auth, dashboard, patients, scans, analysis, reports, analytics,
    settings as settings_router, predict, images,
)

logger = logging.getLogger("retinasense")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    _warn_insecure_config()
    yield


def _warn_insecure_config():
    if not settings.DEBUG and settings.SECRET_KEY == "retinasense-secret-key-change-in-production":
        logger.warning(
            "SECURITY: Running with the default SECRET_KEY. Set a strong "
            "random SECRET_KEY via environment variables in production."
        )


app = FastAPI(title=settings.APP_NAME, version=settings.APP_VERSION, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=500, compresslevel=5)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
    return response


@app.middleware("http")
async def error_handler(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as exc:  # pragma: no cover - safety net
        logger.exception("Unhandled server error on %s %s", request.method, request.url.path)
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})

os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.REPORT_DIR, exist_ok=True)

app.include_router(auth.router)
app.include_router(dashboard.router)
app.include_router(patients.router)
app.include_router(scans.router)
app.include_router(analysis.router)
app.include_router(reports.router)
app.include_router(analytics.router)
app.include_router(settings_router.router)
app.include_router(predict.router)
app.include_router(images.router)


@app.get("/api/health")
async def health():
    return {"status": "ok", "version": settings.APP_VERSION, "model_loaded": _model_available()}


@app.get("/")
async def root():
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "health": "/api/health",
    }


def _model_available() -> bool:
    from pathlib import Path
    return Path(settings.MODEL_PATH).exists()
