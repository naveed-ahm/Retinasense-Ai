from pydantic import BaseModel

class KpiCard(BaseModel):
    title: str
    value: str
    change: str

class ScanActivity(BaseModel):
    day: str
    scans: int
    analyzed: int

class DiseaseDist(BaseModel):
    name: str
    value: int
    color: str

class AiPerformance(BaseModel):
    month: str
    accuracy: float
    f1: float

class ActivityItem(BaseModel):
    time: str
    action: str
    patient: str
    result: str
    type: str

class DashboardResponse(BaseModel):
    kpis: list[KpiCard]
    scan_activity: list[ScanActivity]
    disease_distribution: list[DiseaseDist]
    ai_performance: list[AiPerformance]
    recent_activity: list[ActivityItem]
