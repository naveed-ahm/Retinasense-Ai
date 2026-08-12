import os

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool

from app.config import settings
from app.services.auth_service import get_current_user
from app.models.user import User

router = APIRouter(prefix="/api/images", tags=["Images"])


@router.get("/{name}")
async def get_image(
    name: str,
    current_user: User = Depends(get_current_user),
):
    """Stream an uploaded fundus image or heatmap to an authenticated client."""
    safe = os.path.basename(name)  # block path traversal
    path = os.path.join(settings.UPLOAD_DIR, safe)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Image not found")
    return await run_in_threadpool(FileResponse, path)
