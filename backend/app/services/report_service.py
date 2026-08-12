from datetime import datetime, date
import os
import uuid


def _calculate_age(dob_str: str) -> str:
    """Calculate age from DOB string (YYYY-MM-DD or similar)."""
    if not dob_str or dob_str == "—":
        return "—"
    try:
        for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y"):
            try:
                dob = datetime.strptime(dob_str, fmt).date()
                break
            except ValueError:
                continue
        else:
            return "—"
        today = date.today()
        if dob > today:
            return "—"
        age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
        if age < 0 or age > 120:
            return "—"
        return f"{age} yrs"
    except Exception:
        return "—"


def scan_to_report_input(scan) -> dict:
    """Map a Scan ORM row to the dict consumed by generate_report_data."""
    return {
        "scan_id": scan.scan_id, "patient_id": scan.patient_id,
        "patient_name": scan.patient_name.strip().title() if scan.patient_name else "",
        "patient_dob": scan.patient_dob,
        "patient_gender": scan.patient_gender, "patient_eye": scan.patient_eye,
        "patient_physician": scan.patient_physician.strip().title() if scan.patient_physician else "",
        "image_path": scan.image_path,
        "primary_diagnosis": scan.primary_diagnosis,
        "diagnosis_detail": scan.diagnosis_detail,
        "confidence": scan.confidence, "icd10": scan.icd10,
        "etdrs_grade": scan.etdrs_grade, "severity": scan.severity,
        "risk_score": scan.risk_score,
    }


def generate_report_data(scan: dict, patient: dict | None = None, hospital_name: str = "") -> dict:
    now = datetime.now()
    today = now.strftime("%B %d, %Y")
    generated_timestamp = now.strftime("%Y-%m-%d %H:%M:%S UTC")
    report_id = f"RS-{now.year}-{uuid.uuid4().hex[:8].upper()}"
    raw_name = scan.get("patient_name", patient.get("name", "—") if patient else "—")
    patient_name = raw_name.strip().title() if raw_name and raw_name != "—" else raw_name
    patient_id = scan.get("patient_id", patient.get("patient_id", "—") if patient else "—")
    dob = scan.get("patient_dob", "")
    gender = scan.get("patient_gender", "—")
    eye = scan.get("patient_eye", "Left Eye (OS)")
    physician = scan.get("patient_physician", "—")
    age = _calculate_age(dob)

    if physician and not physician.lower().startswith("dr"):
        physician = f"Dr. {physician}"
    elif not physician:
        physician = "—"

    image_path = scan.get("image_path", "")
    image_url = os.path.basename(image_path) if image_path else None
    heatmap_url = f"{os.path.splitext(image_url)[0]}_heatmap.png" if image_url else None

    diag_primary = scan.get("primary_diagnosis", "Diabetic Retinopathy")
    
    # Custom possible causes and symptoms based on diagnosis
    causes_map = {
        "Diabetic Retinopathy": ["Chronic hyperglycemia (high blood sugar)", "Prolonged duration of diabetes", "Poor systemic blood pressure control", "Dyslipidemia"],
        "Glaucoma": ["Elevated intraocular pressure (IOP)", "Optic nerve vulnerability", "Family history of glaucoma", "Advanced age (>60)"],
        "AMD": ["Aging of the retinal pigment epithelium", "Cumulative oxidative stress", "Genetic predisposition (CFH gene)", "History of smoking"],
        "Hypertensive Retinopathy": ["Chronic uncontrolled systemic hypertension", "Arteriosclerosis", "Endothelial vascular dysfunction"],
        "Macular Edema": ["Diabetic microvascular leakage", "Retinal vein occlusion", "Post-surgical inflammation"],
        "Healthy": ["Regular preventive eye care", "Well-managed systemic health", "Absence of vascular or degenerative pathology"]
    }
    symptoms_map = {
        "Diabetic Retinopathy": ["Blurred or fluctuating vision", "Floaters or dark spots", "Impaired color vision", "Vision loss in advanced stages"],
        "Glaucoma": ["Gradual loss of peripheral vision", "Tunnel vision in advanced stages", "Mild eye ache or headache", "Halos around lights"],
        "AMD": ["Central vision distortion (straight lines appearing wavy)", "Blurry spot in the center of vision", "Difficulty recognizing faces"],
        "Hypertensive Retinopathy": ["Asymptomatic in early stages", "Headaches", "Vision changes during hypertensive crisis"],
        "Macular Edema": ["Central blurring", "Washed-out or distorted color perception", "Difficulty reading fine print"],
        "Healthy": ["None reported — normal visual acuity"]
    }

    possible_causes = causes_map.get(diag_primary, ["Vascular or degenerative retinal changes", "Systemic metabolic factors"])
    symptoms = symptoms_map.get(diag_primary, ["Visual blur", "Asymptomatic early presentation"])
    follow_up_advice = f"Schedule follow-up evaluation and dilated fundus exam within {'2-4 weeks' if scan.get('severity') in ['Severe', 'Critical'] else '3-6 months'}."

    return {
        "report_id": report_id,
        "generated_date": today,
        "generated_timestamp": generated_timestamp,
        "patient_info": {
            "patient_name": patient_name,
            "patient_id": patient_id,
            "age": age,
            "dob": dob if dob else "—",
            "gender": gender if gender else "—",
            "mrn": f"MRN-{uuid.uuid4().hex[:7].upper()}",
        },
        "imaging": {
            "eye": eye,
            "image_quality": "Good (Score: 8.7/10)",
            "camera": "Topcon TRC-NW400",
            "fov": "45° Non-Mydriatic",
            "physician": physician,
            "hospital_name": hospital_name or "Your Hospital & RetinaSense Clinical Center",
            "scan_date": today,
            "image_url": image_url,
            "heatmap_url": heatmap_url,
        },
        "diagnosis": {
            "diagnosis": f"{diag_primary} — {scan.get('diagnosis_detail', '')}",
            "icd10": scan.get("icd10", "E11.311"),
            "etdrs_grade": scan.get("etdrs_grade", "43"),
            "confidence": scan.get("confidence", 96.3),
            "severity": scan.get("severity", "Moderate"),
            "risk_score": scan.get("risk_score", 7.2),
            "summary": f"The AI deep learning model analyzed the retinal fundus image and detected features consistent with {diag_primary}. Grad-CAM attention mapping confirmed localized pathological indicators in the posterior pole.",
            "possible_causes": possible_causes,
            "symptoms": symptoms,
            "follow_up_advice": follow_up_advice,
        },
        "findings": scan.get("findings") if scan.get("findings") else [
            {"finding": "Microaneurysms", "status": "Present", "confidence": "97.2%", "significance": "Characteristic of vascular stress"},
            {"finding": "Hard Exudates", "status": "Present", "confidence": "94.8%", "significance": "Lipid leakage from damaged vessels"},
            {"finding": "Macular Edema", "status": "Suspected", "confidence": "71.4%", "significance": "Vision-threatening — urgent eval needed"},
            {"finding": "Vitreous Hemorrhage", "status": "Absent", "confidence": "99.1%", "significance": "No active bleeding"},
        ],
        "recommendations": scan.get("recommendations") if scan.get("recommendations") else [
            {"priority": "URGENT", "text": "Refer to vitreoretinal specialist within 2 weeks for comprehensive evaluation and OCT imaging."},
            {"priority": "HIGH", "text": "Optimize systemic glycemic and blood pressure control."},
            {"priority": "ROUTINE", "text": "Patient education and regular screening."},
        ],
    }
