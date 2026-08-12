import os
import uuid
from datetime import date

from fastapi import APIRouter, Depends, File, UploadFile, Form, HTTPException
from fastapi.responses import Response
from sqlalchemy import select, Integer
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool

from app.database import get_db
from app.models.scan import Scan
from app.models.patient import Patient
from app.models.prediction import Prediction
from app.models.report import Report
from app.schemas.scan import ScanResponse, ScanStatus
from app.schemas.analysis import AnalysisResponse
from app.schemas.report import ReportResponse
from app.services.ai_service import simulate_ai_analysis, read_upload
from app.services.analysis_service import build_analysis, scan_to_analysis
from app.services.report_service import generate_report_data, scan_to_report_input
from app.services.pdf_service import generate_report_pdf
from app.services.auth_service import get_current_user
from app.models.user import User

router = APIRouter(prefix="/api/scans", tags=["Scans"])


@router.post("", response_model=ScanResponse)
async def create_scan(
    patient_name: str = Form(...),
    patient_id: str = Form(""),
    patient_dob: str = Form(""),
    patient_gender: str = Form(""),
    patient_eye: str = Form("Left Eye (OS)"),
    patient_mobile: str = Form(""),
    patient_notes: str = Form(""),
    image: UploadFile | None = File(None),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    from sqlalchemy import func
    from datetime import datetime

    if not patient_id:
        result = await db.execute(
            select(func.max(func.cast(func.substr(Scan.patient_id, 3), Integer)))
        )
        max_num = result.scalar() or 0
        patient_id = f"P-{max_num + 1:04d}"
    scan_id = f"SC-{uuid.uuid4().hex[:8].upper()}"
    raw_name = f"{current_user.first_name} {current_user.last_name}".strip()
    physician_name = raw_name if not raw_name.lower().startswith("dr.") else raw_name[3:].strip()
    image_path = ""
    heatmap_path = None
    if image:
        image_bytes = await read_upload(image)
        ai_result = await build_analysis(image_bytes, image.filename or "scan.png")
        image_path = ai_result.get("image_path", "")
        heatmap_path = ai_result.get("heatmap_path")
    else:
        ai_result = simulate_ai_analysis()

    age = 0
    if patient_dob:
        try:
            dob_date = datetime.strptime(patient_dob, "%Y-%m-%d").date()
            today = date.today()
            if dob_date > today:
                raise HTTPException(status_code=422, detail="Date of birth cannot be in the future")
            if 0 <= (today.year - dob_date.year) <= 120:
                calc_age = today.year - dob_date.year - ((today.month, today.day) < (dob_date.month, dob_date.day))
                if 0 <= calc_age <= 120:
                    age = calc_age
        except HTTPException:
            raise
        except Exception:
            pass

    patient_name = (patient_name or "").strip().title()
    physician_name = (physician_name or "").strip().title()

    existing = await db.scalar(select(Patient).where(Patient.patient_id == patient_id))
    if existing:
        existing.name = patient_name or existing.name
        existing.age = age or existing.age
        existing.date_of_birth = patient_dob or existing.date_of_birth
        existing.gender = patient_gender or existing.gender
        existing.mobile = patient_mobile or existing.mobile
        existing.physician = physician_name or existing.physician
        existing.condition = ai_result["primary_diagnosis"] or existing.condition
        existing.severity = ai_result["severity"] or existing.severity
        existing.last_scan = datetime.now().strftime("%Y-%m-%d")
        existing.risk = f"{ai_result['risk_score']}/10"
        existing.status = "Active"
        if patient_notes:
            existing.notes = patient_notes
    else:
        db.add(Patient(
            patient_id=patient_id,
            name=patient_name,
            age=age,
            date_of_birth=patient_dob,
            gender=patient_gender,
            mobile=patient_mobile,
            physician=physician_name,
            condition=ai_result["primary_diagnosis"],
            severity=ai_result["severity"],
            last_scan=datetime.now().strftime("%Y-%m-%d"),
            risk=f"{ai_result['risk_score']}/10",
            status="Active",
            notes=patient_notes or "",
        ))

    scan = Scan(
        scan_id=scan_id,
        patient_id=patient_id,
        patient_name=patient_name,
        patient_dob=patient_dob,
        patient_gender=patient_gender,
        patient_eye=patient_eye,
        patient_physician=physician_name,
        patient_notes=patient_notes,
        image_path=image_path,
        status="completed",
        progress=100,
        **{k: ai_result[k] for k in ["primary_diagnosis", "diagnosis_detail", "confidence", "risk_score", "severity", "icd10", "etdrs_grade"]},
        probabilities=ai_result["probabilities"],
        findings=ai_result["findings"],
        recommendations=ai_result["recommendations"],
    )
    db.add(scan)
    prediction = Prediction(
        scan_id=scan_id,
        primary_diagnosis=ai_result["primary_diagnosis"],
        diagnosis_detail=ai_result["diagnosis_detail"],
        confidence=ai_result["confidence"],
        risk_score=ai_result["risk_score"],
        severity=ai_result["severity"],
        icd10=ai_result["icd10"],
        etdrs_grade=ai_result["etdrs_grade"],
        all_probabilities=ai_result["probabilities"],
        findings=ai_result["findings"],
        recommendations=ai_result["recommendations"],
        model_version="efficientnetb3-v1",
    )
    db.add(prediction)
    await db.flush()
    report_data = generate_report_data(scan_to_report_input(scan), hospital_name=current_user.hospital_name or "")
    db.add(Report(
        report_id=report_data["report_id"],
        scan_id=scan_id,
        prediction_id=prediction.id,
        report_data=report_data,
    ))
    await db.commit()
    await db.refresh(scan)
    return ScanResponse(scan_id=scan_id, status="completed", progress=100)





_SORTABLE = {"created_at": Scan.created_at, "confidence": Scan.confidence, "patient_name": Scan.patient_name}


@router.get("", response_model=dict)
async def list_scans(
    search: str = "",
    patient_id: str = "",
    diagnosis: str = "",
    status: str = "",
    severity: str = "",
    sort_by: str = "created_at",
    sort_dir: str = "desc",
    page: int = 1,
    page_size: int = 20,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List scans with search, filtering, sorting and pagination."""
    from sqlalchemy import func

    page = max(page, 1)
    page_size = max(1, min(page_size, 100))
    stmt = select(Scan)
    if search:
        like = f"%{search}%"
        stmt = stmt.where(
            (Scan.patient_name.ilike(like))
            | (Scan.scan_id.ilike(like))
            | (Scan.patient_id.ilike(like))
            | (Scan.primary_diagnosis.ilike(like))
        )
    if patient_id:
        stmt = stmt.where(Scan.patient_id == patient_id)
    if diagnosis:
        stmt = stmt.where(Scan.primary_diagnosis == diagnosis)
    if status:
        stmt = stmt.where(Scan.status == status)
    if severity:
        stmt = stmt.where(Scan.severity == severity)

    base_total = await db.scalar(select(func.count()).select_from(stmt.subquery())) or 0
    filtered = await db.scalar(select(func.count()).select_from(stmt.subquery())) or 0

    col = _SORTABLE.get(sort_by, Scan.created_at)
    stmt = stmt.order_by(col.desc() if sort_dir == "desc" else col.asc())
    stmt = stmt.offset((page - 1) * page_size).limit(page_size)
    result = await db.execute(stmt)
    scans = result.scalars().all()
    return {
        "scans": [
            {
                "scan_id": s.scan_id,
                "patient_id": s.patient_id,
                "patient_name": (s.patient_name or "").strip().title(),
                "primary_diagnosis": s.primary_diagnosis,
                "confidence": s.confidence,
                "severity": s.severity,
                "status": s.status,
                "created_at": s.created_at.isoformat() if s.created_at else None,
                "image_path": os.path.basename(s.image_path) if s.image_path else "",
            }
            for s in scans
        ],
        "total": base_total,
        "filtered": filtered,
        "page": page,
        "page_size": page_size,
    }


@router.get("/{scan_id}/status")
async def get_scan_status(
    scan_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await db.execute(select(Scan).where(Scan.scan_id == scan_id))
    scan = result.scalar_one_or_none()
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found")
    return ScanStatus(scan_id=scan.scan_id, status=scan.status, progress=scan.progress)


@router.get("/{scan_id}/analysis", response_model=AnalysisResponse)
async def get_scan_analysis(
    scan_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await db.execute(select(Scan).where(Scan.scan_id == scan_id))
    scan = result.scalar_one_or_none()
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found")
    return scan_to_analysis(scan)


@router.get("/{scan_id}/report", response_model=ReportResponse)
async def get_scan_report(
    scan_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await db.execute(select(Scan).where(Scan.scan_id == scan_id))
    scan = result.scalar_one_or_none()
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found")
    data = generate_report_data(scan_to_report_input(scan), hospital_name=current_user.hospital_name or "")
    return ReportResponse(**data)


@router.delete("/{scan_id}/report", status_code=204)
async def delete_scan_report(
    scan_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await db.execute(select(Scan).where(Scan.scan_id == scan_id))
    scan = result.scalar_one_or_none()
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found")
    report = await db.scalar(select(Report).where(Report.scan_id == scan_id))
    if report:
        await db.delete(report)
        await db.commit()
    return Response(status_code=204)


@router.delete("/{scan_id}", status_code=204)
async def delete_scan(
    scan_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await db.execute(select(Scan).where(Scan.scan_id == scan_id))
    scan = result.scalar_one_or_none()
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found")
    report = await db.scalar(select(Report).where(Report.scan_id == scan_id))
    if report:
        await db.delete(report)
    prediction = await db.scalar(select(Prediction).where(Prediction.scan_id == scan_id))
    if prediction:
        await db.delete(prediction)
    await db.delete(scan)
    await db.commit()
    return Response(status_code=204)


@router.get("/{scan_id}/report/pdf")
async def get_scan_report_pdf(
    scan_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await db.execute(select(Scan).where(Scan.scan_id == scan_id))
    scan = result.scalar_one_or_none()
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found")
    data = generate_report_data(scan_to_report_input(scan), hospital_name=current_user.hospital_name or "")
    pdf = await run_in_threadpool(generate_report_pdf, data)
    filename = f"{data['report_id']}.pdf"
    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
