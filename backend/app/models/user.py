from sqlalchemy import Column, Integer, String, Boolean, DateTime
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    first_name = Column(String(100), default="Anil")
    last_name = Column(String(100), default="Rajan")
    role = Column(String(100), default="Senior Ophthalmologist")
    institution = Column(String(255), default="Your Hospital")
    hospital_name = Column(String(255), default="Your Hospital & RetinaSense Clinical Center")
    phone = Column(String(50), default="+91 98765 43210")
    specialty = Column(String(100), default="Ophthalmology")
    dark_mode = Column(Boolean, default=False)
    critical_alerts = Column(Boolean, default=True)
    report_ready = Column(Boolean, default=True)
    weekly_digest = Column(Boolean, default=False)
    model_updates = Column(Boolean, default=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    scans = relationship("RetinalScan", back_populates="user", cascade="all, delete-orphan")
    logs = relationship("AuditLog", back_populates="user", cascade="all, delete-orphan")
