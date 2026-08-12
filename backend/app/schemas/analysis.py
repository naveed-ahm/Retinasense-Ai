from pydantic import BaseModel

class PatientSummary(BaseModel):
    patient_id: str
    full_name: str
    dob: str
    gender: str
    eye: str
    physician: str

class DiagnosisResult(BaseModel):
    primary: str
    detail: str
    confidence: float
    risk_score: float
    severity: str
    icd10: str
    etdrs_grade: str

class ProbabilityItem(BaseModel):
    name: str
    pct: float
    color: str

class Finding(BaseModel):
    finding: str
    status: str
    severity: str

class Recommendation(BaseModel):
    icon: str
    title: str
    desc: str
    color: str

class AnalysisResponse(BaseModel):
    patient: PatientSummary
    diagnosis: DiagnosisResult
    probabilities: list[ProbabilityItem]
    findings: list[Finding]
    recommendations: list[Recommendation]
    heatmap_available: bool
    image_url: str | None = None
    heatmap_url: str | None = None
