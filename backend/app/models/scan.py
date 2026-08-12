from sqlalchemy import Column, Integer, String, Float, Text, DateTime, JSON, ForeignKey
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.database import Base


class RetinalScan(Base):
    __tablename__ = "retinal_scans"

    id = Column(Integer, primary_key=True, index=True)
    scan_id = Column(String(50), unique=True, index=True, nullable=False)
    patient_id = Column(String(50), ForeignKey("patients.patient_id", ondelete="CASCADE"), nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    image_path = Column(String(500), default="")
    image_original_name = Column(String(255), default="")
    image_size = Column(Integer, default=0)
    image_format = Column(String(20), default="")
    eye_side = Column(String(20), default="Left Eye (OS)")
    clinical_notes = Column(Text, default="")
    status = Column(String(20), default="pending", index=True)
    progress = Column(Integer, default=0)
    error_message = Column(Text, default="")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    patient = relationship("Patient", back_populates="scans")
    user = relationship("User", back_populates="scans")


class Scan(Base):
    __tablename__ = "scans"

    id = Column(Integer, primary_key=True, index=True)
    scan_id = Column(String(50), unique=True, index=True, nullable=False)
    patient_id = Column(String(50), index=True, nullable=False)
    patient_name = Column(String(255), default="")
    patient_dob = Column(String(20), default="")
    patient_gender = Column(String(20), default="")
    patient_eye = Column(String(20), default="Left Eye (OS)")
    patient_physician = Column(String(255), default="")
    patient_notes = Column(Text, default="")
    image_path = Column(String(500), default="")
    status = Column(String(20), default="completed")
    progress = Column(Integer, default=100)
    primary_diagnosis = Column(String(100), default="Diabetic Retinopathy")
    diagnosis_detail = Column(String(255), default="Stage II — Moderate Non-Proliferative (NPDR)")
    confidence = Column(Float, default=96.3)
    risk_score = Column(Float, default=7.2)
    severity = Column(String(50), default="Moderate")
    icd10 = Column(String(20), default="E11.311")
    etdrs_grade = Column(String(20), default="43")
    probabilities = Column(JSON, default=dict)
    findings = Column(JSON, default=list)
    recommendations = Column(JSON, default=list)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    prediction = relationship("Prediction", back_populates="scan", uselist=False, cascade="all, delete-orphan")
    report = relationship("Report", back_populates="scan", uselist=False, cascade="all, delete-orphan")


class Activity(Base):
    __tablename__ = "activities"

    id = Column(Integer, primary_key=True, index=True)
    time_label = Column(String(20), default="")
    action = Column(String(100), default="")
    patient = Column(String(255), default="")
    result = Column(String(100), default="")
    type = Column(String(20), default="info")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
