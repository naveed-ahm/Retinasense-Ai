from pydantic import BaseModel

class SummaryCard(BaseModel):
    title: str
    value: str
    change: str
    trend: str

class DiseaseTrend(BaseModel):
    month: str
    dr: int
    glaucoma: int
    amd: int

class ModelPerformance(BaseModel):
    month: str
    accuracy: float
    f1: float

class ScanVolume(BaseModel):
    day: str
    scans: int
    analyzed: int

class DetectionStat(BaseModel):
    label: str
    pct: int
    color: str

class CategoryStats(BaseModel):
    title: str
    items: list[DetectionStat]

class AnalyticsResponse(BaseModel):
    summary: list[SummaryCard]
    disease_trends: list[DiseaseTrend]
    model_performance: list[ModelPerformance]
    scan_volume: list[ScanVolume]
    category_stats: list[CategoryStats]
