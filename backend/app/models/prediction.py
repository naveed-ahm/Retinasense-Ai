from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, ForeignKey, Boolean
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.database import Base


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    scan_id = Column(String(50), ForeignKey("scans.scan_id", ondelete="CASCADE"), nullable=False, unique=True, index=True)
    primary_diagnosis = Column(String(100), nullable=False, index=True)
    diagnosis_detail = Column(String(255), default="")
    confidence = Column(Float, nullable=False)
    risk_score = Column(Float, default=0.0)
    severity = Column(String(50), default="", index=True)
    icd10 = Column(String(20), default="")
    etdrs_grade = Column(String(20), default="")
    all_probabilities = Column(JSON, default=dict)
    findings = Column(JSON, default=list)
    recommendations = Column(JSON, default=list)
    model_version = Column(String(50), default="1.0.0")
    inference_time_ms = Column(Float, default=0.0)
    is_final = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    scan = relationship("Scan", back_populates="prediction")
    report = relationship("Report", back_populates="prediction", uselist=False, cascade="all, delete-orphan")
