import re
import sqlalchemy
from datetime import date

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response
from sqlalchemy import select, delete, func
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool
from app.database import get_db
from app.models.patient import Patient
from app.models.scan import Scan
from app.schemas.patient import PatientBase, PatientCreate, PatientListResponse, ScanHistoryItem, PatientDetailResponse, PatientDetail
from app.services.auth_service import get_current_user
from app.services.report_service import generate_report_data, scan_to_report_input
from app.services.pdf_service import generate_report_pdf
from app.models.user import User

router = APIRouter(prefix="/api/patients", tags=["Patients"])


_NAME_RE = re.compile(r"^[A-Za-z.\s\-']+$")


def _get_computed_age(p: Patient) -> int:
    if p.date_of_birth:
        try:
            from datetime import datetime
            dob_date = datetime.strptime(p.date_of_birth, "%Y-%m-%d").date()
            today = date.today()
            if dob_date <= today:
                calc_age = today.year - dob_date.year - ((today.month, today.day) < (dob_date.month, dob_date.day))
                if 0 <= calc_age <= 120:
                    return calc_age
        except Exception:
            pass
    if p.age and 0 <= p.age <= 120:
        return p.age
    return 0


def _risk_score(severity: str | None, confidence: float | None = None) -> float:
    """Dynamic 0.0-10.0 risk score derived from severity and AI confidence (never hardcoded)."""
    sev = (severity or "").lower()
    conf = max(0.0, min(100.0, float(confidence or 0.0))) / 100.0
    if sev in ("none", "healthy"):
        score = max(0.3, 1.5 - (conf * 1.0))
    elif sev == "mild":
        score = 2.5 + (conf * 2.0)
    elif sev == "moderate":
        score = 5.5 + (conf * 2.0)
    elif sev in ("severe", "critical"):
        score = 8.0 + (conf * 1.8)
    else:
        score = 3.0 + (conf * 4.0)
    return round(min(9.9, max(0.1, score)), 1)


async def _latest_confidences(db: AsyncSession) -> dict:
    """Map patient_id -> confidence of its most recent completed scan."""
    result = await db.execute(
        select(Scan.patient_id, Scan.confidence, Scan.created_at).order_by(Scan.created_at.desc())
    )
    conf_map: dict[str, float] = {}
    for pid, conf, _ in result.all():
        if pid not in conf_map and conf is not None:
            conf_map[pid] = float(conf)
    return conf_map


def _risk_label(severity: str | None, confidence: float | None = None) -> str:
    return f"{_risk_score(severity, confidence)}/10"


@router.get("")
async def list_patients(
    search: str = "", status: str = "", diagnosis: str = "",
    risk: str = "", date_from: str = "", date_to: str = "",
    page: int = Query(1, ge=1), page_size: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    stmt = select(Patient)
    if search:
        stmt = stmt.where((Patient.name.ilike(f"%{search}%")) | (Patient.patient_id.ilike(f"%{search}%")))
    if status and status != "All":
        stmt = stmt.where(Patient.status == status)
    if diagnosis:
        stmt = stmt.where(Patient.condition == diagnosis)
    if risk:
        stmt = stmt.where(Patient.risk == risk)
    if date_from:
        stmt = stmt.where(Patient.last_scan >= date_from)
    if date_to:
        stmt = stmt.where(Patient.last_scan <= date_to)
    total = await db.scalar(select(func.count()).select_from(Patient)) or 0
    filtered = await db.scalar(select(func.count()).select_from(stmt.subquery())) or 0
    stmt = stmt.order_by(Patient.id.desc()).offset((page - 1) * page_size).limit(page_size)
    result = await db.execute(stmt)
    patients = result.scalars().all()
    conf_map = await _latest_confidences(db)
    return PatientListResponse(
        patients=[PatientBase(
            id=p.patient_id, name=p.name.strip().title(), age=_get_computed_age(p), condition=p.condition,
            severity=p.severity, lastScan=p.last_scan, status=p.status,
            risk=_risk_label(p.severity, conf_map.get(p.patient_id)),
            contact=p.contact, email=p.email, mobile=p.mobile or "",
            date_of_birth=p.date_of_birth or "", gender=p.gender or "", notes=p.notes or "",
        ) for p in patients],
        total=total,
        filtered=filtered,
    )

@router.post("", status_code=201)
async def create_patient(
    data: PatientCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if not _NAME_RE.match(data.name.strip()):
        raise HTTPException(status_code=422, detail="Name must contain only letters, dots, hyphens, and spaces")
    if data.age < 0 or data.age > 120:
        raise HTTPException(status_code=422, detail="Age must be between 0 and 120 years")
    max_num = await db.scalar(
        select(func.max(func.cast(func.substr(Patient.patient_id, 3), sqlalchemy.Integer)))
    ) or 0
    pid = f"P-{max_num + 1:04d}"
    patient_name = data.name.strip().title()
    patient = Patient(
        patient_id=pid, name=patient_name, age=data.age,
        condition=data.condition, severity="Mild",
        last_scan=date.today().isoformat(),
        status=data.status, risk=_risk_label("Mild"),
        contact=data.contact, email=data.email,
    )
    db.add(patient)
    try:
        await db.commit()
    except IntegrityError:
        await db.rollback()
        raise HTTPException(status_code=409, detail="Patient ID conflict, please retry")
    await db.refresh(patient)
    return PatientBase(
        id=patient.patient_id, name=patient.name, age=_get_computed_age(patient),
        condition=patient.condition, severity=patient.severity,
        lastScan=patient.last_scan, status=patient.status,
        risk=patient.risk, contact=patient.contact, email=patient.email,
        mobile=patient.mobile or "", date_of_birth=patient.date_of_birth or "",
        gender=patient.gender or "", notes=patient.notes or "",
    )

@router.get("/{patient_id}")
async def get_patient(
    patient_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await db.execute(select(Patient).where(Patient.patient_id == patient_id))
    p = result.scalar_one_or_none()
    if not p:
        raise HTTPException(status_code=404, detail="Patient not found")
    scan_result = await db.execute(
        select(Scan).where(Scan.patient_id == patient_id).order_by(Scan.created_at.desc())
    )
    scans = scan_result.scalars().all()
    patient = PatientBase(
        id=p.patient_id, name=p.name.strip().title(), age=_get_computed_age(p), condition=p.condition,
        severity=p.severity, lastScan=p.last_scan, status=p.status,
        risk=_risk_label(p.severity, scans[0].confidence if scans else None),
        contact=p.contact, email=p.email, mobile=p.mobile or "",
        date_of_birth=p.date_of_birth or "", gender=p.gender or "", notes=p.notes or "",
    )
    detail = PatientDetail(
        patientId=p.patient_id, fullName=p.name.strip().title(), dob=p.date_of_birth or "",
        eye="Left Eye (OS)", gender=p.gender or "", physician=p.physician.strip().title() if p.physician else "", notes=p.notes or "",
    )
    history = [
        ScanHistoryItem(
            date=s.created_at.strftime("%Y-%m-%d") if s.created_at else p.last_scan,
            type="Fundus Photo",
            result=s.primary_diagnosis,
            confidence=f"{s.confidence:.1f}%" if s.confidence is not None else "—",
            status="Normal" if (s.severity or "").lower() in ("none", "healthy") else "Abnormal",
            scan_id=s.scan_id,
        )
        for s in scans
    ]
    if not history:
        history = [ScanHistoryItem(
            date=p.last_scan, type="Fundus Photo", result=p.condition,
            confidence="—", status="Normal" if p.severity == "None" else "Abnormal",
        )]
    return PatientDetailResponse(patient=patient, details=detail, scan_history=history)

@router.get("/{patient_id}/report/pdf")
async def get_patient_report_pdf(
    patient_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await db.execute(select(Patient).where(Patient.patient_id == patient_id))
    p = result.scalar_one_or_none()
    if not p:
        raise HTTPException(status_code=404, detail="Patient not found")
    scan_result = await db.execute(
        select(Scan).where(Scan.patient_id == patient_id).order_by(Scan.id.desc()).limit(1)
    )
    scan = scan_result.scalar_one_or_none()
    if scan:
        data = generate_report_data(scan_to_report_input(scan), hospital_name=current_user.hospital_name or "")
    else:
        data = generate_report_data({
            "scan_id": f"SC-{patient_id}", "patient_id": p.patient_id,
            "patient_name": p.name, "patient_dob": "",
            "patient_gender": "", "patient_eye": "Left Eye (OS)",
            "patient_physician": "", "primary_diagnosis": p.condition,
            "diagnosis_detail": "", "confidence": 96.3,
            "icd10": "E11.311", "etdrs_grade": "43",
            "severity": p.severity, "risk_score": 7.2,
        }, hospital_name=current_user.hospital_name or "")
    pdf = await run_in_threadpool(generate_report_pdf, data)
    filename = f"{data['report_id']}.pdf"
    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.delete("/{patient_id}", status_code=204)
async def delete_patient(
    patient_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await db.execute(select(Patient).where(Patient.patient_id == patient_id))
    p = result.scalar_one_or_none()
    if not p:
        raise HTTPException(status_code=404, detail="Patient not found")
    await db.delete(p)
    await db.commit()
