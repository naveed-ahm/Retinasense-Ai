from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.database import Base


class Patient(Base):
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(String(50), unique=True, index=True, nullable=False)
    name = Column(String(255), nullable=False, index=True)
    age = Column(Integer, nullable=False)
    date_of_birth = Column(String(20), default="")
    gender = Column(String(20), default="")
    physician = Column(String(255), default="")
    condition = Column(String(100), default="", index=True)
    severity = Column(String(50), default="")
    last_scan = Column(String(20), default="")
    status = Column(String(50), default="Active", index=True)
    risk = Column(String(20), default="", index=True)
    contact = Column(String(50), default="")
    mobile = Column(String(50), default="")
    email = Column(String(255), default="")
    notes = Column(String(1000), default="")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    scans = relationship("RetinalScan", back_populates="patient", cascade="all, delete-orphan")
