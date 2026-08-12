from pydantic import BaseModel

class PatientInfo(BaseModel):
    patient_name: str
    patient_id: str
    age: str = "—"
    dob: str
    gender: str
    mrn: str

class ImagingDetail(BaseModel):
    eye: str
    image_quality: str
    camera: str
    fov: str
    physician: str
    hospital_name: str = "Apollo Hospitals & RetinaSense Clinical Center"
    scan_date: str = "—"
    image_url: str | None = None
    heatmap_url: str | None = None

class ReportDiagnosis(BaseModel):
    diagnosis: str
    icd10: str
    etdrs_grade: str
    confidence: float
    severity: str = "Moderate"
    risk_score: float = 0.0
    summary: str
    possible_causes: list[str] = []
    symptoms: list[str] = []
    follow_up_advice: str = ""

class ReportFinding(BaseModel):
    finding: str
    status: str
    confidence: str
    significance: str

class ReportRecommendation(BaseModel):
    priority: str
    text: str

class ReportResponse(BaseModel):
    report_id: str
    generated_date: str
    generated_timestamp: str
    patient_info: PatientInfo
    imaging: ImagingDetail
    diagnosis: ReportDiagnosis
    findings: list[ReportFinding]
    recommendations: list[ReportRecommendation]
