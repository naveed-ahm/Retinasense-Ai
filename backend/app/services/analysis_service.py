"""
Shared AI analysis builder used by the scans and predict endpoints.
"""
import logging
import os
import uuid

import cv2
import numpy as np
from starlette.concurrency import run_in_threadpool

from app.ai.inference import ModelLoader
from app.services.ai_service import save_upload
from app.schemas.analysis import AnalysisResponse, PatientSummary, DiagnosisResult, ProbabilityItem, Finding, Recommendation
from app.config import settings

_log = logging.getLogger("retinasense.analysis")

_model_loader = None


def _get_model() -> ModelLoader:
    global _model_loader
    if _model_loader is None:
        _model_loader = ModelLoader()
    return _model_loader


DETAIL_MAP = {
    "Diabetic Retinopathy": "Stage II — Moderate Non-Proliferative (NPDR)",
    "Glaucoma": "Open-Angle Glaucoma — Elevated IOP",
    "AMD": "Age-related Macular Degeneration — Dry Type",
    "Hypertensive Retinopathy": "Grade II — Mild to Moderate",
    "Macular Edema": "Clinically Significant Macular Edema (CSME)",
    "Healthy": "Normal Retina — No abnormalities detected",
}
ICD10_MAP = {
    "Diabetic Retinopathy": "E11.311",
    "Glaucoma": "H40.10X0",
    "AMD": "H35.30",
    "Hypertensive Retinopathy": "I11.9",
    "Macular Edema": "H35.81",
    "Healthy": "Z01.00",
}
COLOR_MAP = {
    "Diabetic Retinopathy": "bg-blue-500",
    "Glaucoma": "bg-violet-400",
    "AMD": "bg-sky-400",
    "Hypertensive Retinopathy": "bg-indigo-400",
    "Macular Edema": "bg-amber-400",
    "Healthy": "bg-emerald-400",
}


def _save_heatmap(overlay: np.ndarray, image_path: str) -> str:
    """Persist a Grad-CAM overlay next to the uploaded image."""
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    base = os.path.splitext(os.path.basename(image_path))[0]
    heat_path = os.path.join(settings.UPLOAD_DIR, f"{base}_heatmap.png")
    cv2.imwrite(heat_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    return heat_path


async def build_analysis(image_bytes: bytes, filename: str) -> dict:
    """Decode, validate, run inference and build the full analysis payload.

    Returns an analysis dict including `image_path` and `heatmap_path`
    (paths to the persisted originals). Raises HTTPException(400) when the
    bytes do not form a decodable image.
    """
    np_arr = np.frombuffer(image_bytes, np.uint8)
    if np_arr.size == 0:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="File is not a valid image")
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="File is not a valid image")

    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    loader = _get_model()
    result = await run_in_threadpool(loader.predict, image_rgb)

    probs = result.get("all_probabilities", {})
    probabilities = [
        {"name": k, "pct": round(v * 100, 1), "color": COLOR_MAP.get(k, "bg-gray-400")}
        for k, v in sorted(probs.items(), key=lambda x: x[1], reverse=True)
    ]
    diagnosis = result.get("diagnosis", "Unknown")
    confidence = result.get("confidence", 0.0)
    severity = result.get("severity", "Unknown")
    risk_score = calculate_dynamic_risk_score(severity, confidence, diagnosis)

    image_path = save_upload(image_bytes, filename)
    heatmap_path = None
    if result.get("overlay") is not None:
        heatmap_path = _save_heatmap(result["overlay"], image_path)

    return {
        "primary_diagnosis": diagnosis,
        "diagnosis_detail": DETAIL_MAP.get(diagnosis, diagnosis),
        "confidence": confidence,
        "risk_score": risk_score,
        "severity": severity,
        "icd10": ICD10_MAP.get(diagnosis, "Z01.00"),
        "etdrs_grade": "43",
        "probabilities": probabilities,
        "findings": _generate_findings(diagnosis),
        "recommendations": _generate_recommendations(diagnosis, severity),
        "heatmap_available": heatmap_path is not None,
        "image_path": image_path,
        "heatmap_path": heatmap_path,
        "is_mock": result.get("is_mock", False),
    }


def calculate_dynamic_risk_score(severity: str, confidence: float, diagnosis: str = "") -> float:
    """Calculate a dynamic 0.0-10.0 risk score based on disease severity, diagnosis, and AI confidence."""
    sev = (severity or "").lower()
    diag = (diagnosis or "").lower()
    conf = max(0.0, min(100.0, float(confidence or 0.0))) / 100.0

    if sev in ("none", "healthy") or diag == "healthy":
        score = max(0.3, round(1.5 - (conf * 1.0), 1))
    elif sev == "mild":
        score = round(2.5 + (conf * 2.0), 1)
    elif sev == "moderate":
        score = round(5.5 + (conf * 2.0), 1)
    elif sev in ("severe", "critical"):
        score = round(8.0 + (conf * 1.8), 1)
    else:
        score = round(3.0 + (conf * 4.0), 1)
    return min(9.9, max(0.1, score))


def scan_to_analysis(scan) -> AnalysisResponse:
    """Build an AnalysisResponse from a stored Scan row."""
    image_url = None
    heatmap_url = None
    if scan.image_path:
        base = os.path.basename(scan.image_path)
        image_url = base
        heat_path = os.path.join(settings.UPLOAD_DIR, f"{os.path.splitext(base)[0]}_heatmap.png")
        if os.path.isfile(heat_path):
            heatmap_url = os.path.basename(heat_path)
    return AnalysisResponse(
        patient=PatientSummary(
            patient_id=scan.patient_id, full_name=scan.patient_name,
            dob=scan.patient_dob, gender=scan.patient_gender,
            eye=scan.patient_eye, physician=scan.patient_physician,
        ),
        diagnosis=DiagnosisResult(
            primary=scan.primary_diagnosis, detail=scan.diagnosis_detail,
            confidence=scan.confidence, risk_score=scan.risk_score,
            severity=scan.severity, icd10=scan.icd10,
            etdrs_grade=scan.etdrs_grade,
        ),
        probabilities=[ProbabilityItem(**p) for p in (scan.probabilities or [])],
        findings=[Finding(**f) for f in (scan.findings or [])],
        recommendations=[Recommendation(**r) for r in (scan.recommendations or [])],
        heatmap_available=heatmap_url is not None,
        image_url=image_url,
        heatmap_url=heatmap_url,
    )


def _generate_findings(diagnosis: str) -> list:
    common = [
        {"finding": "Microaneurysms", "status": "Not detected", "severity": "success"},
        {"finding": "Hard Exudates", "status": "Not detected", "severity": "success"},
        {"finding": "Vitreous Hemorrhage", "status": "Not detected", "severity": "success"},
        {"finding": "Neovascularization", "status": "Not detected", "severity": "success"},
        {"finding": "Cotton-wool spots", "status": "Not detected", "severity": "success"},
        {"finding": "Macular Edema", "status": "Not detected", "severity": "success"},
    ]
    per_disease = {
        "Diabetic Retinopathy": [
            {"finding": "Microaneurysms", "status": "Detected", "severity": "warning"},
            {"finding": "Hard Exudates", "status": "Detected", "severity": "warning"},
            {"finding": "Vitreous Hemorrhage", "status": "Not detected", "severity": "success"},
            {"finding": "Neovascularization", "status": "Not detected", "severity": "success"},
            {"finding": "Cotton-wool spots", "status": "Detected", "severity": "warning"},
            {"finding": "Macular Edema", "status": "Suspected", "severity": "error"},
        ],
        "Glaucoma": [
            {"finding": "Cup-to-disc ratio", "status": "Elevated (>0.6)", "severity": "warning"},
            {"finding": "Retinal nerve fiber layer", "status": "Thinning", "severity": "warning"},
            {"finding": "Visual field defect", "status": "Consistent with glaucoma", "severity": "warning"},
            {"finding": "IOP", "status": "Elevated (24 mmHg)", "severity": "error"},
            {"finding": "Optic disc hemorrhage", "status": "Not detected", "severity": "success"},
            {"finding": "Peripapillary atrophy", "status": "Present", "severity": "warning"},
        ],
        "AMD": [
            {"finding": "Drusen", "status": "Detected (multiple soft)", "severity": "warning"},
            {"finding": "Geographic atrophy", "status": "Not detected", "severity": "success"},
            {"finding": "Pigmentary changes", "status": "Present", "severity": "warning"},
            {"finding": "Choroidal neovascularization", "status": "Not detected", "severity": "success"},
            {"finding": "Subretinal fluid", "status": "Not detected", "severity": "success"},
            {"finding": "RPE detachment", "status": "Not detected", "severity": "success"},
        ],
        "Hypertensive Retinopathy": [
            {"finding": "Arteriolar narrowing", "status": "Present (A-V ratio 1:2)", "severity": "warning"},
            {"finding": "AV nicking", "status": "Detected", "severity": "warning"},
            {"finding": "Copper wiring", "status": "Present", "severity": "warning"},
            {"finding": "Flame hemorrhages", "status": "Not detected", "severity": "success"},
            {"finding": "Cotton-wool spots", "status": "Detected", "severity": "warning"},
            {"finding": "Papilledema", "status": "Not detected", "severity": "success"},
        ],
        "Macular Edema": [
            {"finding": "Macular thickening", "status": "Present", "severity": "error"},
            {"finding": "Cystoid spaces", "status": "Detected", "severity": "error"},
            {"finding": "Hard exudates", "status": "Present", "severity": "warning"},
            {"finding": "Subretinal fluid", "status": "Suspected", "severity": "warning"},
            {"finding": "Loss of foveal depression", "status": "Present", "severity": "warning"},
            {"finding": "Vitreomacular traction", "status": "Not detected", "severity": "success"},
        ],
    }
    return per_disease.get(diagnosis, common)


def _generate_recommendations(diagnosis: str, severity: str) -> list:
    urgent = severity in ("Severe", "Critical")
    recs = [
        {"icon": "Calendar", "title": "Follow-up Schedule",
         "desc": "Repeat fundus exam in %s months." % ("1-3" if urgent else "3-6"),
         "color": "text-blue-600 bg-blue-50"},
        {"icon": "HeartPulse", "title": "Systemic Management",
         "desc": "Optimize blood pressure < 130/80 mmHg. Review glycemic control if diabetic.",
         "color": "text-emerald-600 bg-emerald-50"},
    ]
    if urgent:
        recs.append({"icon": "Microscope", "title": "Urgent Referral",
                     "desc": "Immediate retinal specialist review recommended.",
                     "color": "text-red-600 bg-red-50"})
    else:
        recs.append({"icon": "Microscope", "title": "Specialist Referral",
                     "desc": "Consider retinal specialist review if symptoms progress.",
                     "color": "text-amber-600 bg-amber-50"})
    return recs
