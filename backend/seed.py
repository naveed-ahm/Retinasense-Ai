"""Seed the database with initial data matching frontend mock data."""
import asyncio
from app.database import init_db, async_session
from app.models.user import User
from app.models.patient import Patient
from app.models.scan import Scan, Activity
from app.utils.security import hash_password

INITIAL_PATIENTS = [
    {"patient_id": "P-0041", "name": "Eleanor Vasquez", "age": 67, "condition": "Diabetic Retinopathy", "severity": "Moderate", "last_scan": "2024-07-22", "status": "Active", "risk": "High", "contact": "+1 (555) 234-5678", "email": "e.vasquez@email.com"},
    {"patient_id": "P-0042", "name": "Marcus Chen", "age": 54, "condition": "Glaucoma", "severity": "Mild", "last_scan": "2024-07-21", "status": "Active", "risk": "Medium", "contact": "+1 (555) 345-6789", "email": "m.chen@email.com"},
    {"patient_id": "P-0043", "name": "Aisha Patel", "age": 72, "condition": "AMD", "severity": "Severe", "last_scan": "2024-07-20", "status": "Critical", "risk": "High", "contact": "+91 98765 11111", "email": "a.patel@email.com"},
    {"patient_id": "P-0044", "name": "Robert Kim", "age": 61, "condition": "Healthy", "severity": "None", "last_scan": "2024-07-19", "status": "Stable", "risk": "Low", "contact": "+1 (555) 456-7890", "email": "r.kim@email.com"},
    {"patient_id": "P-0045", "name": "Sarah Johnson", "age": 58, "condition": "Diabetic Retinopathy", "severity": "Mild", "last_scan": "2024-07-18", "status": "Active", "risk": "Medium", "contact": "+1 (555) 567-8901", "email": "s.johnson@email.com"},
    {"patient_id": "P-0046", "name": "Thomas Nguyen", "age": 45, "condition": "Healthy", "severity": "None", "last_scan": "2024-07-17", "status": "Stable", "risk": "Low", "contact": "+1 (555) 678-9012", "email": "t.nguyen@email.com"},
    {"patient_id": "P-0047", "name": "Linda Foster", "age": 69, "condition": "Glaucoma", "severity": "Moderate", "last_scan": "2024-07-16", "status": "Active", "risk": "High", "contact": "+1 (555) 789-0123", "email": "l.foster@email.com"},
    {"patient_id": "P-0048", "name": "James Martinez", "age": 76, "condition": "AMD", "severity": "Mild", "last_scan": "2024-07-15", "status": "Active", "risk": "Medium", "contact": "+1 (555) 890-1234", "email": "j.martinez@email.com"},
]

SEED_SCAN = {
    "scan_id": "SC-0001",
    "patient_id": "P-0041",
    "patient_name": "Eleanor Vasquez",
    "patient_dob": "1957-03-12",
    "patient_gender": "Female",
    "patient_eye": "Left Eye (OS)",
    "patient_physician": "Dr. Rajan",
    "patient_notes": "History of diabetes, HbA1c 8.2%",
    "image_path": "",
    "status": "completed",
    "progress": 100,
    "primary_diagnosis": "Diabetic Retinopathy",
    "diagnosis_detail": "Stage II — Moderate Non-Proliferative (NPDR)",
    "confidence": 96.3,
    "risk_score": 7.2,
    "severity": "Moderate",
    "icd10": "E11.311",
    "etdrs_grade": "43",
    "probabilities": [
        {"name": "Diabetic Retinopathy", "pct": 96.3, "color": "bg-blue-500"},
        {"name": "Hypertensive Retinopathy", "pct": 23.1, "color": "bg-indigo-400"},
        {"name": "Glaucoma", "pct": 12.7, "color": "bg-violet-400"},
        {"name": "Age-related Macular Deg.", "pct": 8.4, "color": "bg-sky-400"},
        {"name": "Healthy Retina", "pct": 3.7, "color": "bg-emerald-400"},
    ],
    "findings": [
        {"finding": "Microaneurysms", "status": "Detected", "severity": "warning"},
        {"finding": "Hard Exudates", "status": "Detected", "severity": "warning"},
        {"finding": "Vitreous Hemorrhage", "status": "Not detected", "severity": "success"},
        {"finding": "Neovascularization", "status": "Not detected", "severity": "success"},
        {"finding": "Cotton-wool spots", "status": "Detected", "severity": "warning"},
        {"finding": "Macular Edema", "status": "Suspected", "severity": "error"},
    ],
    "recommendations": [
        {"icon": "Calendar", "title": "Follow-up Schedule", "desc": "Repeat fundus exam in 3-6 months. Consider fluorescein angiography to assess macular perfusion.", "color": "text-blue-600 bg-blue-50"},
        {"icon": "HeartPulse", "title": "Glycemic Control", "desc": "Refer to endocrinology. Target HbA1c < 7%. Optimize blood pressure < 130/80 mmHg.", "color": "text-emerald-600 bg-emerald-50"},
        {"icon": "Microscope", "title": "Specialist Referral", "desc": "Urgent retinal specialist review recommended. Possible laser photocoagulation therapy indicated.", "color": "text-amber-600 bg-amber-50"},
    ],
}

SEED_ACTIVITIES = [
    {"time_label": "2m ago", "action": "Scan completed", "patient": "Eleanor Vasquez", "result": "DR Detected", "type": "warning"},
    {"time_label": "14m ago", "action": "Report generated", "patient": "Marcus Chen", "result": "Glaucoma Stage I", "type": "info"},
    {"time_label": "31m ago", "action": "New patient added", "patient": "Diana Okonkwo", "result": "Intake complete", "type": "success"},
    {"time_label": "1h ago", "action": "Critical alert", "patient": "Aisha Patel", "result": "AMD Severe — urgent", "type": "error"},
    {"time_label": "2h ago", "action": "Scan completed", "patient": "Robert Kim", "result": "Healthy retina", "type": "success"},
]

async def seed():
    await init_db()
    async with async_session() as session:
        from sqlalchemy import select, func

        user_count = await session.scalar(select(func.count(User.id)))
        if user_count == 0:
            user = User(
                email="dr.rajan@apollo.org",
                password_hash=hash_password("demo1234"),
                first_name="Anil",
                last_name="Rajan",
                role="Senior Ophthalmologist",
                institution="Apollo Hospitals",
                phone="+91 98765 43210",
                specialty="Ophthalmology",
            )
            session.add(user)

        patient_count = await session.scalar(select(func.count(Patient.id)))
        if patient_count == 0:
            for p in INITIAL_PATIENTS:
                session.add(Patient(**p))

        scan_count = await session.scalar(select(func.count(Scan.id)))
        if scan_count == 0:
            session.add(Scan(**SEED_SCAN))

        activity_count = await session.scalar(select(func.count(Activity.id)))
        if activity_count == 0:
            for a in SEED_ACTIVITIES:
                session.add(Activity(**a))

        await session.commit()
        print("Database seeded successfully!")

if __name__ == "__main__":
    asyncio.run(seed())
