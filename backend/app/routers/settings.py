from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from app.database import get_db
from app.models.user import User
from app.schemas.settings import ProfileUpdate, PasswordChange, NotificationPrefs, ThemeUpdate
from app.services.auth_service import get_current_user
from app.utils.security import verify_password, hash_password

router = APIRouter(prefix="/api/settings", tags=["Settings"])

@router.get("/profile")
async def get_profile(current_user: User = Depends(get_current_user)):
    initials = "".join(w[0] for w in f"{current_user.first_name} {current_user.last_name}".split() if w).upper()
    return {
        "first_name": current_user.first_name,
        "last_name": current_user.last_name,
        "email": current_user.email,
        "phone": current_user.phone,
        "specialty": current_user.specialty,
        "institution": current_user.institution,
        "hospital_name": current_user.hospital_name or "Your Hospital & RetinaSense Clinical Center",
        "role": current_user.role,
        "initials": initials,
    }

@router.put("/profile")
async def update_profile(
    data: ProfileUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    email = data.email.strip().lower()
    existing = await db.scalar(select(User).where(User.email == email, User.id != current_user.id))
    if existing:
        raise HTTPException(status_code=409, detail="Email is already in use by another account")
    current_user.first_name = data.first_name.strip()
    current_user.last_name = data.last_name.strip()
    current_user.email = email
    current_user.phone = data.phone.strip()
    current_user.specialty = data.specialty.strip()
    current_user.institution = data.institution.strip()
    current_user.hospital_name = data.hospital_name.strip() if data.hospital_name else current_user.hospital_name
    try:
        await db.commit()
    except IntegrityError:
        await db.rollback()
        raise HTTPException(status_code=409, detail="Email is already in use by another account")
    return {"message": "Profile saved successfully"}

@router.put("/password")
async def change_password(
    data: PasswordChange,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if len(data.new_password) < 8:
        raise HTTPException(status_code=400, detail="New password must be at least 8 characters")
    if data.new_password == data.current_password:
        raise HTTPException(status_code=400, detail="New password must differ from the current password")
    if not verify_password(data.current_password, current_user.password_hash):
        raise HTTPException(status_code=400, detail="Current password is incorrect")
    current_user.password_hash = hash_password(data.new_password)
    await db.commit()
    return {"message": "Password updated successfully"}

@router.get("/notifications")
async def get_notifications(current_user: User = Depends(get_current_user)):
    return {
        "critical_alerts": current_user.critical_alerts,
        "report_ready": current_user.report_ready,
        "weekly_digest": current_user.weekly_digest,
        "model_updates": current_user.model_updates,
    }

@router.put("/notifications")
async def update_notifications(
    data: NotificationPrefs,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    current_user.critical_alerts = data.critical_alerts
    current_user.report_ready = data.report_ready
    current_user.weekly_digest = data.weekly_digest
    current_user.model_updates = data.model_updates
    await db.commit()
    return {"message": "Notification preferences updated"}

@router.put("/theme")
async def update_theme(
    data: ThemeUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    current_user.dark_mode = data.dark_mode
    await db.commit()
    return {"message": "Theme updated successfully"}
