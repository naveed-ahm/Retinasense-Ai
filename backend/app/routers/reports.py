from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool

from app.database import get_db
from app.models.scan import Scan
from app.schemas.report import ReportResponse
from app.services.report_service import generate_report_data, scan_to_report_input
from app.services.pdf_service import generate_report_pdf
from app.services.auth_service import get_current_user
from app.models.user import User

router = APIRouter(prefix="/api/reports", tags=["Reports"])

_DEFAULT_SCAN = {
    "scan_id": "SC-DEFAULT", "patient_id": "P-0041",
    "patient_name": "Eleanor Vasquez", "patient_dob": "1957-03-12",
    "patient_gender": "Female", "patient_eye": "Left Eye (OS)",
    "patient_physician": "Dr. Rajan",
    "primary_diagnosis": "Diabetic Retinopathy",
    "diagnosis_detail": "Stage II — Moderate Non-Proliferative (NPDR)",
    "confidence": 96.3, "icd10": "E11.311",
    "etdrs_grade": "43", "severity": "Moderate", "risk_score": 7.2,
}


async def _latest_scan_data(db: AsyncSession) -> dict:
    result = await db.execute(select(Scan).order_by(Scan.id.desc()).limit(1))
    scan = result.scalar_one_or_none()
    return scan_to_report_input(scan) if scan else _DEFAULT_SCAN


@router.get("/latest")
async def get_latest_report(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    data = generate_report_data(await _latest_scan_data(db), hospital_name=current_user.hospital_name or "")
    return ReportResponse(**data)


@router.get("/latest/pdf")
async def get_latest_report_pdf(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    data = generate_report_data(await _latest_scan_data(db), hospital_name=current_user.hospital_name or "")
    pdf = await run_in_threadpool(generate_report_pdf, data)
    filename = f"{data['report_id']}.pdf"
    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
