from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.database import get_db
from app.models.scan import Scan
from app.services.analysis_service import scan_to_analysis
from app.services.auth_service import get_current_user
from app.models.user import User

router = APIRouter(prefix="/api/analysis", tags=["Analysis"])


@router.get("/latest")
async def get_latest_analysis(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await db.execute(select(Scan).order_by(Scan.id.desc()).limit(1))
    scan = result.scalar_one_or_none()
    if not scan:
        raise HTTPException(status_code=404, detail="No scans available")
    return scan_to_analysis(scan)
