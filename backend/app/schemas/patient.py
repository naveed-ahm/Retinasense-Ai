import re
from pydantic import BaseModel, field_validator
from typing import Optional

_NAME_RE = re.compile(r"^[A-Za-z]+(?: [A-Za-z]+)*$")

class PatientBase(BaseModel):
    id: str
    name: str
    age: int
    condition: str
    severity: str
    lastScan: str
    status: str
    risk: str
    contact: str
    email: str
    mobile: str = ""
    date_of_birth: str = ""
    gender: str = ""
    notes: str = ""

class PatientCreate(BaseModel):
    name: str
    age: int
    contact: str = ""
    email: str = ""
    condition: str = "Diabetic Retinopathy"
    status: str = "Active"

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Name is required")
        if not _NAME_RE.match(v):
            raise ValueError("Name must contain only letters and spaces")
        return v

class PatientListResponse(BaseModel):
    patients: list[PatientBase]
    total: int
    filtered: int

class ScanHistoryItem(BaseModel):
    date: str
    type: str
    result: str
    confidence: str
    status: str
    scan_id: str = ""

class PatientDetail(BaseModel):
    patientId: str
    fullName: str
    dob: str
    eye: str
    gender: str
    physician: str
    notes: str

class PatientDetailResponse(BaseModel):
    patient: PatientBase | None
    details: PatientDetail | None
    scan_history: list[ScanHistoryItem]
