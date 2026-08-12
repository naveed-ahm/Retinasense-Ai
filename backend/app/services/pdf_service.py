from io import BytesIO
import os

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    Image,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from app.config import settings

NAVY = colors.HexColor("#0F2B46")
TEAL = colors.HexColor("#0E7C86")
SLATE = colors.HexColor("#5A6B7B")
LIGHT = colors.HexColor("#EEF3F7")
WHITE = colors.white

RED = colors.HexColor("#C62828")
ORANGE = colors.HexColor("#EF6C00")
GREEN = colors.HexColor("#2E7D32")

PAGE_W, PAGE_H = A4
TOP_MARGIN = 38 * mm
BOTTOM_MARGIN = 26 * mm
CONTENT_W = PAGE_W - 2 * 15 * mm

_BODY = ParagraphStyle(
    name="Body",
    fontName="Helvetica",
    fontSize=9.5,
    leading=13.5,
    textColor=colors.HexColor("#1C2B36"),
    spaceAfter=5,
)
_BODY_BOLD = ParagraphStyle(
    name="BodyBold",
    parent=_BODY,
    fontName="Helvetica-Bold",
)
_SECTION = ParagraphStyle(
    name="Section",
    fontName="Helvetica-Bold",
    fontSize=11,
    leading=14,
    textColor=NAVY,
    spaceBefore=10,
    spaceAfter=5,
)
_KV_KEY = ParagraphStyle(
    name="KvKey",
    fontName="Helvetica-Bold",
    fontSize=8,
    leading=10,
    textColor=SLATE,
)
_KV_VAL = ParagraphStyle(
    name="KvVal",
    fontName="Helvetica",
    fontSize=9.5,
    leading=11.5,
    textColor=colors.HexColor("#1C2B36"),
)
_CELL = ParagraphStyle(
    name="Cell",
    fontName="Helvetica",
    fontSize=8.5,
    leading=11,
    textColor=colors.HexColor("#1C2B36"),
)
_CELL_BOLD = ParagraphStyle(
    name="CellBold",
    parent=_CELL,
    fontName="Helvetica-Bold",
)
_CELL_WHITE = ParagraphStyle(
    name="CellWhite",
    parent=_CELL,
    fontName="Helvetica-Bold",
    textColor=WHITE,
)
_STATUS_STYLE = ParagraphStyle(
    name="Status",
    parent=_CELL_BOLD,
)
_SUBTLE = ParagraphStyle(
    name="Subtle",
    fontName="Helvetica-Oblique",
    fontSize=7.5,
    leading=10,
    textColor=SLATE,
)
_IMG_CAP = ParagraphStyle(
    name="ImgCap",
    fontName="Helvetica-Bold",
    fontSize=8.5,
    leading=10,
    textColor=NAVY,
    alignment=TA_CENTER,
)


def generate_report_pdf(data: dict) -> bytes:
    buf = BytesIO()

    def on_page(canvas, doc):
        _draw_header(canvas, doc, data)
        _draw_footer(canvas, doc, data)

    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        topMargin=TOP_MARGIN,
        bottomMargin=BOTTOM_MARGIN,
        leftMargin=15 * mm,
        rightMargin=15 * mm,
        title="RetinaSense AI Clinical Report",
        author="RetinaSense AI",
    )
    doc.build(_build_story(data), onFirstPage=on_page, onLaterPages=on_page)
    return buf.getvalue()


def _draw_header(canvas, doc, data: dict):
    canvas.saveState()
    canvas.setFillColor(NAVY)
    canvas.rect(0, PAGE_H - 30 * mm, PAGE_W, 30 * mm, stroke=0, fill=1)
    canvas.setFillColor(TEAL)
    canvas.rect(0, PAGE_H - 32 * mm, PAGE_W, 2 * mm, stroke=0, fill=1)

    canvas.setFillColor(WHITE)
    canvas.setFont("Helvetica-Bold", 14)
    canvas.drawString(15 * mm, PAGE_H - 16 * mm, "RetinaSense AI")
    canvas.setFont("Helvetica", 8.5)
    canvas.drawString(15 * mm, PAGE_H - 22 * mm, "Clinical Retinal Diagnosis Report")

    canvas.setFont("Helvetica-Bold", 9)
    canvas.drawRightString(PAGE_W - 15 * mm, PAGE_H - 16 * mm, data.get("report_id", ""))
    canvas.setFont("Helvetica", 8)
    canvas.drawRightString(PAGE_W - 15 * mm, PAGE_H - 22 * mm, data.get("generated_date", ""))
    canvas.restoreState()


def _draw_footer(canvas, doc, data: dict):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#D5DEE6"))
    canvas.setLineWidth(0.5)
    canvas.line(15 * mm, 20 * mm, PAGE_W - 15 * mm, 20 * mm)

    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(SLATE)
    canvas.drawString(
        15 * mm, 14 * mm,
        "This AI-generated report is for clinical decision support only and does not replace a "
        "specialist's diagnosis. Verify all findings during the patient consultation.",
    )
    canvas.drawString(15 * mm, 9 * mm, data.get("generated_timestamp", ""))
    canvas.setFont("Helvetica-Bold", 7)
    canvas.drawRightString(PAGE_W - 15 * mm, 9 * mm, f"Page {doc.page}")
    canvas.restoreState()


def _build_story(data: dict) -> list:
    patient = data.get("patient_info", {})
    imaging = data.get("imaging", {})
    diag = data.get("diagnosis", {})
    story = []

    story.append(_severity_banner(diag))

    story.append(Paragraph("Patient Information", _SECTION))
    story.append(_kv_grid([
        ("Patient Name", patient.get("patient_name", "—")),
        ("Patient ID", patient.get("patient_id", "—")),
        ("MRN", patient.get("mrn", "—")),
        ("Date of Birth", patient.get("dob", "—")),
        ("Age", patient.get("age", "—")),
        ("Gender", patient.get("gender", "—")),
    ]))
    story.append(Spacer(1, 4))

    story.append(Paragraph("Imaging Details", _SECTION))
    story.append(_kv_grid([
        ("Eye Examined", imaging.get("eye", "—")),
        ("Image Quality", imaging.get("image_quality", "—")),
        ("Camera", imaging.get("camera", "—")),
        ("Field of View", imaging.get("fov", "—")),
        ("Examining Physician", imaging.get("physician", "—")),
        ("Scan Date", imaging.get("scan_date", "—")),
    ]))
    story.append(Spacer(1, 4))

    story.append(Paragraph("Diagnosis", _SECTION))
    story.append(_diagnosis_box(diag))

    story.append(Paragraph("AI Clinical Summary", _SECTION))
    story.append(Paragraph(diag.get("summary", ""), _BODY))

    causes = diag.get("possible_causes", [])
    symptoms = diag.get("symptoms", [])
    if causes or symptoms:
        story.append(Paragraph("Supporting Information", _SECTION))
        story.append(_two_column_lists("Possible Causes", causes, "Patient-Reported Symptoms", symptoms))
        story.append(Spacer(1, 4))

    findings = data.get("findings", [])
    if findings:
        story.append(Paragraph("Clinical Findings", _SECTION))
        story.append(_findings_table(findings))

    if imaging.get("image_url"):
        story.append(Paragraph("Imaging Evidence", _SECTION))
        story.append(_images_row(imaging))

    recommendations = data.get("recommendations", [])
    if recommendations:
        story.append(Paragraph("Recommendations", _SECTION))
        story.append(_recommendations_table(recommendations))

    follow_up = diag.get("follow_up_advice", "")
    if follow_up:
        box = Table(
            [[Paragraph(f"<b>Follow-up:</b> {follow_up}", _BODY)]],
            colWidths=[CONTENT_W],
        )
        box.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#E7F1F2")),
            ("BOX", (0, 0), (-1, -1), 0.75, TEAL),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ]))
        story.append(Spacer(1, 8))
        story.append(box)

    return story


def _severity_banner(diag: dict) -> Table:
    severity = diag.get("severity", "Moderate")
    color = {"Critical": RED, "Severe": ORANGE, "Moderate": ORANGE}.get(severity, GREEN)
    left = Table(
        [[Paragraph(f"<b>{severity} Severity</b>", _cell(WHITE, 11))]],
        colWidths=[62 * mm],
    )
    left.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), color),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    right = Table(
        [[Paragraph(
            f"<font color='white' size='20'><b>{diag.get('confidence', 0):.1f}%</b></font>"
            f"<br/><font color='white' size='8'>AI CONFIDENCE</font>",
            _cell(WHITE, 9),
        )]],
        colWidths=[62 * mm],
    )
    right.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), NAVY),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    outer = Table([[left, right]], colWidths=[62 * mm, 62 * mm])
    outer.setStyle(TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 0),
                               ("RIGHTPADDING", (0, 0), (-1, -1), 0)]))
    return outer


def _diagnosis_box(diag: dict) -> Table:
    risk = diag.get("risk_score")
    risk_txt = f"Risk Score: {risk:.1f}/10" if isinstance(risk, (int, float)) else ""
    inner = [
        [Paragraph(f"<b>{diag.get('diagnosis', '—')}</b>", _cell(NAVY, 12))],
        [Paragraph(
            f"<font size='8.5' color='#5A6B7B'>ICD-10: <b>{diag.get('icd10', '—')}</b>"
            f"&nbsp;&nbsp;|&nbsp;&nbsp;ETDRS Grade: <b>{diag.get('etdrs_grade', '—')}</b>"
            f"&nbsp;&nbsp;|&nbsp;&nbsp;{risk_txt}</font>",
            _BODY,
        )],
    ]
    box = Table(inner, colWidths=[CONTENT_W])
    box.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT),
        ("BOX", (0, 0), (-1, -1), 0.75, colors.HexColor("#C7D4DE")),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    return box


def _kv_grid(pairs: list[tuple[str, str]]) -> Table:
    rows = []
    for i in range(0, len(pairs), 2):
        row = []
        for key, val in pairs[i:i + 2]:
            row += [
                Paragraph(key, _KV_KEY),
                Paragraph(val if val else "—", _KV_VAL),
            ]
        rows.append(row)
    col_w = CONTENT_W / 4
    t = Table(rows, colWidths=[col_w] * 4)
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("LINEBELOW", (0, 0), (-1, -2), 0.4, colors.HexColor("#E4EBF1")),
    ]))
    return t


def _two_column_lists(left_title, left_items, right_title, right_items) -> Table:
    def col(title, items):
        parts = [Paragraph(title, _BODY_BOLD)]
        for it in items:
            parts.append(Paragraph(f"•&nbsp; {it}", _BODY))
        return parts

    t = Table(
        [[col(left_title, left_items), col(right_title, right_items)]],
        colWidths=[CONTENT_W / 2, CONTENT_W / 2],
    )
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 2),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ]))
    return t


def _findings_table(findings: list[dict]) -> Table:
    header = [Paragraph(f"<b>{h}</b>", _cell(WHITE, 8.5)) for h in
              ["Finding", "Status", "Confidence", "Significance"]]
    status_color = {
        "Present": RED,
        "Suspected": ORANGE,
        "Absent": GREEN,
    }
    rows = [header]
    for f in findings:
        status = f.get("status", "")
        color = status_color.get(status, SLATE)
        rows.append([
            Paragraph(f.get("finding", ""), _CELL_BOLD),
            Paragraph(f"<font color='#{color.hexval()[2:]}'><b>{status}</b></font>", _CELL_BOLD),
            Paragraph(f.get("confidence", ""), _CELL),
            Paragraph(f.get("significance", ""), _CELL),
        ])
    col_w = [38 * mm, 26 * mm, 22 * mm, CONTENT_W - 86 * mm]
    t = Table(rows, colWidths=col_w, repeatRows=1)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("LINEBELOW", (0, 1), (-1, -2), 0.4, colors.HexColor("#E4EBF1")),
        ("BOX", (0, 0), (-1, -1), 0.75, colors.HexColor("#C7D4DE")),
        ("GRID", (0, 1), (-1, -1), 0.2, colors.HexColor("#E4EBF1")),
    ]
    for i, row in enumerate(rows[1:], start=1):
        if i % 2 == 0:
            style.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#F7FAFC")))
    t.setStyle(TableStyle(style))
    return t


def _images_row(imaging: dict) -> Table:
    def cell(name, cap):
        path = _resolve_image(name)
        if not path:
            return [Paragraph(f"<i>{name}</i> (image unavailable)", _SUBTLE)]
        try:
            img = ImageReader(path)
            iw, ih = img.getSize()
            box = 76 * mm
            if iw and ih:
                scale = min(box / iw, box / ih)
                w, h = iw * scale, ih * scale
            else:
                w, h = box, box
            img_flow = Image(path, width=w, height=h)
        except Exception:
            return [Paragraph(f"<i>{name}</i> (could not embed)", _SUBTLE)]
        return [img_flow, Spacer(1, 3), Paragraph(cap, _IMG_CAP)]

    left = cell(imaging.get("image_url"), "Original Fundus Image")
    right = cell(imaging.get("heatmap_url"), "Grad-CAM Attention Map")
    inner = Table(
        [[left, right]],
        colWidths=[CONTENT_W / 2, CONTENT_W / 2],
    )
    inner.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]))
    return inner


def _recommendations_table(recommendations: list[dict]) -> Table:
    priority_style = {
        "URGENT": (RED, WHITE),
        "HIGH": (ORANGE, WHITE),
        "ROUTINE": (GREEN, WHITE),
    }
    rows = []
    for rec in recommendations:
        priority = rec.get("priority", "ROUTINE").upper()
        bg, fg = priority_style.get(priority, (SLATE, WHITE))
        badge = Paragraph(
            f"<font color='#{fg.hexval()[2:]}'><b>{priority}</b></font>",
            _cell(fg, 8),
        )
        badge_cell = Table([[badge]], colWidths=[22 * mm])
        badge_cell.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), bg),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        rows.append([badge_cell, Paragraph(rec.get("text", ""), _CELL)])
    t = Table(rows, colWidths=[26 * mm, CONTENT_W - 26 * mm])
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 2),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


def _resolve_image(name: str) -> str | None:
    if not name:
        return None
    path = os.path.join(settings.UPLOAD_DIR, os.path.basename(name))
    return path if os.path.isfile(path) else None


def _cell(color, size=9.5) -> ParagraphStyle:
    return ParagraphStyle(
        name=f"cell_{color.hexval()[2:]}_{size}",
        fontName="Helvetica-Bold",
        fontSize=size,
        leading=size + 2,
        textColor=color,
        alignment=TA_CENTER,
    )
