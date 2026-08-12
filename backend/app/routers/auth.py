from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.database import get_db
from app.models.user import User
from app.schemas.auth import LoginRequest, LoginResponse, AuthUser, ForgotPasswordRequest, RegisterRequest, RequestAccessRequest
from app.utils.security import verify_password, hash_password, create_access_token
from app.utils.rate_limit import check_rate_limit

router = APIRouter(prefix="/api/auth", tags=["Auth"])

_LOGIN_LIMIT = 5
_LOGIN_WINDOW = 60


@router.post("/register")
async def register(req: RegisterRequest, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.email == req.email))
    if result.scalar_one_or_none():
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="An account with this email already exists")
    user = User(
        email=req.email,
        password_hash=hash_password(req.password),
        first_name=req.first_name,
        last_name=req.last_name,
    )
    db.add(user)
    await db.commit()
    return {"message": "Account created successfully"}


@router.post("/login")
async def login(req: LoginRequest, request: Request, db: AsyncSession = Depends(get_db)):
    client_ip = request.client.host if request.client else "unknown"
    if not check_rate_limit(f"login:{client_ip}", _LOGIN_LIMIT, _LOGIN_WINDOW):
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                            detail="Too many login attempts. Please try again later.")
    result = await db.execute(select(User).where(User.email == req.email))
    user = result.scalar_one_or_none()
    if not user or not verify_password(req.password, user.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password")
    if not getattr(user, "is_active", True):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Account is disabled")
    token = create_access_token({"sub": user.email})
    initials = "".join(w[0] for w in f"{user.first_name} {user.last_name}".split() if w).upper()
    return LoginResponse(
        token=token,
        user=AuthUser(
            name=f"{user.first_name} {user.last_name}",
            role=user.role,
            initials=initials,
            email=user.email,
        )
    )

@router.post("/forgot-password")
async def forgot_password(req: ForgotPasswordRequest):
    return {"message": "If an account exists, a reset link has been sent."}

@router.post("/request-access")
async def request_access(req: RequestAccessRequest):
    return {"message": "Request submitted successfully."}
