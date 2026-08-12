from fastapi import APIRouter, Depends, File, UploadFile, Form, HTTPException

from app.services.analysis_service import build_analysis
from app.services.ai_service import save_upload, read_upload
from app.services.auth_service import get_current_user
from app.models.user import User

router = APIRouter(prefix="/api", tags=["Prediction"])


def _public_url(path: str) -> str:
    from os.path import basename
    return f"/api/images/{basename(path)}"


@router.post("/predict")
async def predict(
    image: UploadFile = File(...),
    patient_id: str = Form(""),
    patient_name: str = Form(""),
    current_user: User = Depends(get_current_user),
):
    """One-shot inference on a single fundus image. Does not persist a scan."""
    image_bytes = await read_upload(image)
    result = await build_analysis(image_bytes, image.filename or "scan.png")
    return {
        "primary_diagnosis": result["primary_diagnosis"],
        "diagnosis_detail": result["diagnosis_detail"],
        "confidence": result["confidence"],
        "severity": result["severity"],
        "risk_score": result["risk_score"],
        "icd10": result["icd10"],
        "etdrs_grade": result["etdrs_grade"],
        "probabilities": result["probabilities"],
        "findings": result["findings"],
        "recommendations": result["recommendations"],
        "heatmap_available": result["heatmap_available"],
        "is_mock": result.get("is_mock", False),
        "image_url": _public_url(result["image_path"]),
        "heatmap_url": _public_url(result["heatmap_path"]) if result.get("heatmap_path") else None,
        "patient_id": patient_id,
        "patient_name": patient_name,
    }


@router.post("/upload")
async def upload_image(
    image: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
):
    """Validate and store an uploaded fundus image. Returns its public URL."""
    image_bytes = await read_upload(image)
    path = save_upload(image_bytes, image.filename or "scan.png")
    return {
        "file_name": path.split("\\")[-1].split("/")[-1],
        "image_url": _public_url(path),
        "size_bytes": len(image_bytes),
    }
