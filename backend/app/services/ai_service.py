from PIL import Image
import numpy as np
import io, os, uuid

from fastapi import HTTPException

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10 MB
ALLOWED_CONTENT_TYPES = {
    "image/jpeg", "image/png", "image/bmp", "image/tiff",
    "image/webp", "application/octet-stream",
}

def validate_upload(filename: str, data: bytes, content_type: str | None = None) -> str:
    """Validate an uploaded image file. Returns the safe extension to use."""
    if not data:
        raise HTTPException(status_code=400, detail="Empty file uploaded")
    if len(data) > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"Image too large. Maximum allowed size is {MAX_UPLOAD_SIZE // (1024 * 1024)} MB",
        )
    ext = os.path.splitext(filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext or '(none)'}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )
    if content_type and content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail=f"Unsupported content type '{content_type}'")
    if _sniff_image(data) is None:
        raise HTTPException(status_code=400, detail="File is not a valid image")
    return ext

def _sniff_image(data: bytes) -> np.ndarray | None:
    """Decode the image bytes to verify it is a real, decodable image."""
    import cv2
    np_arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    return img

def save_upload(image_bytes: bytes, filename: str) -> str:
    """Validate and persist an uploaded image. Returns absolute path."""
    from app.config import settings
    ext = validate_upload(filename, image_bytes)
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    name = f"{uuid.uuid4().hex}{ext}"
    path = os.path.join(settings.UPLOAD_DIR, name)
    with open(path, "wb") as f:
        f.write(image_bytes)
    return path


async def read_upload(upload) -> bytes:
    """Read an UploadFile in chunks, rejecting files larger than MAX_UPLOAD_SIZE.

    Streams so a malicious client cannot exhaust server memory before the
    size check runs.
    """
    chunks = []
    total = 0
    while True:
        chunk = await upload.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > MAX_UPLOAD_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"Image too large. Maximum allowed size is {MAX_UPLOAD_SIZE // (1024 * 1024)} MB",
            )
        chunks.append(chunk)
    if not chunks:
        raise HTTPException(status_code=400, detail="Empty file uploaded")
    return b"".join(chunks)

def simulate_ai_analysis(image_bytes: bytes | None = None, filename: str | None = None) -> dict:
    probabilities = [
        {"name": "Diabetic Retinopathy", "pct": 96.3, "color": "bg-blue-500"},
        {"name": "Hypertensive Retinopathy", "pct": 23.1, "color": "bg-indigo-400"},
        {"name": "Glaucoma", "pct": 12.7, "color": "bg-violet-400"},
        {"name": "Age-related Macular Deg.", "pct": 8.4, "color": "bg-sky-400"},
        {"name": "Healthy Retina", "pct": 3.7, "color": "bg-emerald-400"},
    ]
    findings = [
        {"finding": "Microaneurysms", "status": "Detected", "severity": "warning"},
        {"finding": "Hard Exudates", "status": "Detected", "severity": "warning"},
        {"finding": "Vitreous Hemorrhage", "status": "Not detected", "severity": "success"},
        {"finding": "Neovascularization", "status": "Not detected", "severity": "success"},
        {"finding": "Cotton-wool spots", "status": "Detected", "severity": "warning"},
        {"finding": "Macular Edema", "status": "Suspected", "severity": "error"},
    ]
    recommendations = [
        {"icon": "Calendar", "title": "Follow-up Schedule", "desc": "Repeat fundus exam in 3-6 months. Consider fluorescein angiography to assess macular perfusion.", "color": "text-blue-600 bg-blue-50"},
        {"icon": "HeartPulse", "title": "Glycemic Control", "desc": "Refer to endocrinology. Target HbA1c < 7%. Optimize blood pressure < 130/80 mmHg.", "color": "text-emerald-600 bg-emerald-50"},
        {"icon": "Microscope", "title": "Specialist Referral", "desc": "Urgent retinal specialist review recommended. Possible laser photocoagulation therapy indicated.", "color": "text-amber-600 bg-amber-50"},
    ]
    return {
        "primary_diagnosis": "Diabetic Retinopathy",
        "diagnosis_detail": "Stage II — Moderate Non-Proliferative (NPDR)",
        "confidence": 96.3,
        "risk_score": 7.2,
        "severity": "Moderate",
        "icd10": "E11.311",
        "etdrs_grade": "43",
        "probabilities": probabilities,
        "findings": findings,
        "recommendations": recommendations,
        "heatmap_available": True,
    }
