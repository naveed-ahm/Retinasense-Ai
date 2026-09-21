"""Independent, optional MATLAB image-analysis endpoint.

This router only reads images already saved by the established upload flow. It
does not call the disease model, alter preprocessing, persist database data, or
affect Grad-CAM/PDF generation.
"""
from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Depends, Form, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool

from app.config import settings
from app.database import get_db
from app.models.scan import Scan
from app.models.user import User
from app.services.ai_service import ALLOWED_EXTENSIONS
from app.services.auth_service import get_current_user
try:
    from matlab_service import matlab_runner
except ImportError:
    from backend.matlab_service import matlab_runner

router = APIRouter(prefix="/api/matlab", tags=["MATLAB Add-on"])


def _uploaded_image(stored_path: str) -> Path:
    """Resolve a database-stored upload strictly inside the established upload directory."""
    if not stored_path:
        raise HTTPException(status_code=400, detail="This scan has no uploaded image")
    candidate = Path(stored_path).resolve()
    upload_root = Path(settings.UPLOAD_DIR).resolve()
    if candidate.parent != upload_root or candidate.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Invalid uploaded image reference")
    if not candidate.is_file():
        raise HTTPException(status_code=404, detail="Uploaded image not found")
    return candidate


@router.post("/analyze")
async def analyze_matlab_image(
    scan_id: str = Form(...),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Run optional visualization/quality analysis on a previously uploaded image."""
    scan = await db.scalar(select(Scan).where(Scan.scan_id == scan_id))
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found")
    image_path = _uploaded_image(scan.image_path)
    if not matlab_runner.is_matlab_available():
        return {"matlab_available": False, "matlab_status": "MATLAB analysis unavailable"}

    output_path = Path(settings.UPLOAD_DIR) / f"{image_path.stem}_matlab_{uuid4().hex}.png"
    try:
        quality, enhancement = await run_in_threadpool(
            lambda: (
                matlab_runner.run_quality_analysis(image_path),
                matlab_runner.run_enhancement(image_path, output_path),
            )
        )
    except Exception:
        # An add-on failure must be non-fatal and never affect the AI workflow.
        return {"matlab_available": False, "matlab_status": "MATLAB analysis unavailable"}

    response = {
        "matlab_available": True,
        "matlab_status": "MATLAB analysis complete",
        "quality": quality,
        "enhanced_image": Path(enhancement["enhancement_path"]).name,
    }
    try:
        # Quantitative measurements are intentionally optional for this prototype.
        response["features"] = await run_in_threadpool(matlab_runner.run_retinal_features, image_path)
    except Exception:
        pass
    return response
