from datetime import datetime, timedelta, timezone

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.scan import Scan
from app.models.patient import Patient
from app.models.report import Report

_UTC = timezone.utc


def utcnow() -> datetime:
    """UTC now. Strips tzinfo for naive comparison (SQLite) but keeps it for aware (PostgreSQL)."""
    return datetime.now(_UTC)


def _to_naive(dt: datetime) -> datetime:
    """Strip tzinfo for safe subtraction across SQLite/PostgreSQL."""
    return dt.replace(tzinfo=None) if dt.tzinfo else dt

DISEASE_COLORS = {
    "Diabetic Retinopathy": "#F59E0B",
    "Glaucoma": "#8B5CF6",
    "AMD": "#06B6D4",
    "Macular Edema": "#F97316",
    "Hypertensive Retinopathy": "#EC4899",
    "Healthy": "#10B981",
}
SEVERITY_COLORS = {
    "Critical": "bg-red-500",
    "Severe": "bg-orange-500",
    "Moderate": "bg-amber-500",
    "Mild": "bg-emerald-500",
    "None": "bg-green-500",
}
DEFAULT_DISEASE_COLOR = "#94A3B8"
DEFAULT_SEVERITY_COLOR = "bg-slate-400"

_WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


async def _count(db, model):
    return await db.scalar(select(func.count(model.id))) or 0


async def _recent_scans(db, days: int):
    cutoff = _to_naive(utcnow() - timedelta(days=days))
    result = await db.execute(select(Scan).where(Scan.created_at >= cutoff))
    return result.scalars().all()


async def compute_dashboard(db: AsyncSession) -> dict:
    total_scans = await _count(db, Scan)
    total_patients = await _count(db, Patient)
    reports_count = await _count(db, Report)
    avg_conf = await db.scalar(select(func.avg(Scan.confidence))) or 0.0

    week_ago = _to_naive(utcnow() - timedelta(days=7))
    scans_7d = await db.scalar(
        select(func.count(Scan.id)).where(Scan.created_at >= week_ago)
    ) or 0
    reports_7d = await db.scalar(
        select(func.count(Report.id)).where(Report.generated_at >= week_ago)
    ) or 0

    activity = await _weekly_activity(db, days=7)
    distribution = await _disease_distribution(db)
    performance = await _monthly_performance(db, months=7)
    recent = await _recent_activity(db, limit=6)

    return {
        "kpis": [
            {"title": "Total Scans", "value": f"{total_scans:,}", "change": f"+{scans_7d} this week"},
            {"title": "Avg Confidence", "value": f"{avg_conf:.1f}%" if avg_conf else "—", "change": f"{total_scans} total scans" if total_scans else "No scans yet"},
            {"title": "Active Patients", "value": f"{total_patients:,}", "change": "Registered patients"},
            {"title": "Reports Generated", "value": f"{reports_count:,}", "change": f"+{reports_7d} this week"},
        ],
        "scan_activity": activity,
        "disease_distribution": distribution,
        "ai_performance": performance,
        "recent_activity": recent,
    }


async def compute_analytics(db: AsyncSession) -> dict:
    total_scans = await _count(db, Scan)
    total_patients = await _count(db, Patient)
    reports_count = await _count(db, Report)
    avg_conf = await db.scalar(select(func.avg(Scan.confidence))) or 0.0

    month_ago = _to_naive(utcnow() - timedelta(days=30))
    scans_month = await db.scalar(
        select(func.count(Scan.id)).where(Scan.created_at >= month_ago)
    ) or 0

    return {
        "summary": [
            {"title": "Scans This Month", "value": f"{scans_month:,}", "change": f"{total_scans:,} total", "trend": "up" if scans_month else "flat"},
            {"title": "Avg Confidence", "value": f"{avg_conf:.1f}%" if avg_conf else "—", "change": "Across all scans", "trend": "up"},
            {"title": "Total Patients", "value": f"{total_patients:,}", "change": f"{reports_count:,} reports", "trend": "up"},
            {"title": "Reports Generated", "value": f"{reports_count:,}", "change": "Persisted reports", "trend": "up"},
        ],
        "disease_trends": await _disease_trends(db, months=6),
        "model_performance": await _monthly_performance(db, months=7),
        "scan_volume": await _weekly_activity(db, days=7),
        "category_stats": await _category_stats(db),
    }


async def _weekly_activity(db: AsyncSession, days: int) -> list[dict]:
    now = utcnow()
    rows = await _recent_scans(db, days)
    buckets = {}
    for i in range(days - 1, -1, -1):
        d = (now - timedelta(days=i)).date()
        buckets[d] = {"scans": 0, "analyzed": 0}
    for s in rows:
        key = s.created_at.date() if s.created_at else None
        if key and key in buckets:
            buckets[key]["scans"] += 1
            if s.status == "completed":
                buckets[key]["analyzed"] += 1
    result = []
    for i in range(days - 1, -1, -1):
        d = (now - timedelta(days=i)).date()
        label = _WEEKDAYS[d.weekday()]
        result.append({"day": label, **buckets[d]})
    return result


async def _disease_distribution(db: AsyncSession) -> list[dict]:
    rows = (await db.execute(select(Scan.primary_diagnosis, func.count(Scan.id)).group_by(Scan.primary_diagnosis))).all()
    if not rows or sum(c for _, c in rows if c) == 0:
        rows = (await db.execute(select(Patient.condition, func.count(Patient.id)).group_by(Patient.condition))).all()
    
    valid_rows = [(name.strip() if name else "Unknown", count) for name, count in rows if count]
    total = sum(c for _, c in valid_rows)
    if total == 0:
        return []

    items = []
    pct_sum = 0
    for idx, (name, count) in enumerate(valid_rows):
        if idx == len(valid_rows) - 1:
            pct = round(100.0 - pct_sum, 1)
        else:
            pct = round((count / total) * 100.0, 1)
            pct_sum += pct
        items.append({
            "name": name,
            "value": pct,
            "color": DISEASE_COLORS.get(name, DEFAULT_DISEASE_COLOR),
        })
    return items


async def _monthly_performance(db: AsyncSession, months: int) -> list[dict]:
    now = utcnow()
    rows = (await db.execute(
        select(Scan.created_at, Scan.confidence).where(Scan.created_at >= now - timedelta(days=months * 31))
    )).all()
    buckets = {}
    for i in range(months - 1, -1, -1):
        m = now.month - i
        y = now.year
        while m <= 0:
            m += 12
            y -= 1
        buckets[(y, m)] = {"sum": 0.0, "n": 0}
    for created, conf in rows:
        if created and conf is not None:
            key = (created.year, created.month)
            if key in buckets:
                buckets[key]["sum"] += conf
                buckets[key]["n"] += 1
    result = []
    for i in range(months - 1, -1, -1):
        m = now.month - i
        y = now.year
        while m <= 0:
            m += 12
            y -= 1
        b = buckets[(y, m)]
        acc = (b["sum"] / b["n"]) if b["n"] else 0.0
        result.append({"month": _MONTHS[m - 1], "accuracy": round(acc, 1), "f1": round(acc, 1)})
    return result


async def _disease_trends(db: AsyncSession, months: int) -> list[dict]:
    now = utcnow()
    rows = (await db.execute(
        select(Scan.created_at, Scan.primary_diagnosis).where(
            Scan.created_at >= now - timedelta(days=months * 31)
        )
    )).all()
    buckets = {}
    for i in range(months - 1, -1, -1):
        m = now.month - i
        y = now.year
        while m <= 0:
            m += 12
            y -= 1
        buckets[(y, m)] = {"dr": 0, "glaucoma": 0, "amd": 0}
    for created, diag in rows:
        if not created:
            continue
        key = (created.year, created.month)
        if key not in buckets:
            continue
        if diag and "diabetic" in diag.lower():
            buckets[key]["dr"] += 1
        elif diag and "glaucoma" in diag.lower():
            buckets[key]["glaucoma"] += 1
        elif diag and "amd" in diag.lower():
            buckets[key]["amd"] += 1
    result = []
    for i in range(months - 1, -1, -1):
        m = now.month - i
        y = now.year
        while m <= 0:
            m += 12
            y -= 1
        result.append({"month": _MONTHS[m - 1], **buckets[(y, m)]})
    return result


async def _category_stats(db: AsyncSession) -> list[dict]:
    diag_rows = (await db.execute(
        select(Scan.primary_diagnosis, func.count(Scan.id)).group_by(Scan.primary_diagnosis)
    )).all()
    total_diag = sum(c for _, c in diag_rows) or 1
    diag_items = [
        {"label": name or "Unknown", "pct": round(count * 100 / total_diag), "color": "bg-blue-500"}
        for name, count in diag_rows
    ]

    sev_rows = (await db.execute(
        select(Scan.severity, func.count(Scan.id)).group_by(Scan.severity)
    )).all()
    total_sev = sum(c for _, c in sev_rows) or 1
    sev_items = [
        {"label": sev or "Unknown", "pct": round(count * 100 / total_sev), "color": SEVERITY_COLORS.get(sev, DEFAULT_SEVERITY_COLOR)}
        for sev, count in sev_rows
    ]

    age_rows = (await db.execute(select(Patient.age, func.count(Patient.id)).group_by(Patient.age))).all()
    age_buckets = {"35-50": 0, "50-65": 0, "65-80": 0, "80+": 0}
    for age, count in age_rows:
        if age is None:
            continue
        if age < 35:
            continue
        elif age < 50:
            age_buckets["35-50"] += count
        elif age < 65:
            age_buckets["50-65"] += count
        elif age < 80:
            age_buckets["65-80"] += count
        else:
            age_buckets["80+"] += count
    total_age = sum(age_buckets.values()) or 1
    age_items = [
        {"label": label, "pct": round(count * 100 / total_age), "color": "bg-indigo-500"}
        for label, count in age_buckets.items()
    ]

    return [
        {"title": "Top Detected Conditions", "items": diag_items[:4]},
        {"title": "Detection by Severity", "items": sev_items[:4]},
        {"title": "Patient Demographics", "items": age_items},
    ]


async def _recent_activity(db: AsyncSession, limit: int) -> list[dict]:
    result = await db.execute(select(Scan).order_by(Scan.created_at.desc()).limit(limit))
    scans = result.scalars().all()
    items = []
    now = utcnow()
    for s in scans:
        ts = s.created_at if s.created_at else now
        delta = _to_naive(now) - _to_naive(ts)
        if delta.days > 0:
            time_str = f"{delta.days}d ago"
        elif delta.seconds // 3600 > 0:
            time_str = f"{delta.seconds // 3600}h ago"
        elif delta.seconds // 60 > 0:
            time_str = f"{delta.seconds // 60}m ago"
        else:
            time_str = "just now"
        severity = (s.severity or "").lower()
        if severity in ("severe", "critical"):
            itype = "error"
        elif severity in ("moderate", "mild"):
            itype = "warning"
        else:
            itype = "success"
        items.append({
            "time": time_str,
            "action": "Scan completed",
            "patient": (s.patient_name or s.patient_id).strip().title(),
            "result": s.primary_diagnosis or "—",
            "type": itype,
        })
    return items
