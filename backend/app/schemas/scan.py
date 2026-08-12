from pydantic import BaseModel
from typing import Optional

class ScanCreate(BaseModel):
    patient_id: str
    patient_name: str
    patient_dob: str = ""
    patient_gender: str = ""
    patient_eye: str = "Left Eye (OS)"
    patient_physician: str = ""
    patient_notes: str = ""

class ScanResponse(BaseModel):
    scan_id: str
    status: str
    progress: int

class ScanStatus(BaseModel):
    scan_id: str
    status: str
    progress: int
