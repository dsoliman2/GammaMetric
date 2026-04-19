"""
GammaMetric PDF Report Generator
Produces a 7-page reliability report matching GammaMetric_Reliability_Report_DEMO.pdf.
"""

from __future__ import annotations
from datetime import datetime
from io import BytesIO
from typing import Optional

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer,
    Table, TableStyle, HRFlowable, KeepTogether, PageBreak,
)

from core.sensitivity_engine import (
    SensitivityInput, SensitivityResult, compute,
    normalize_kernel, SLICE_THICKNESS_RANGE_MM, CTDIVOL_RANGE_MGY,
)

# ---------------------------------------------------------------------------
# Brand colors
# ---------------------------------------------------------------------------
GM_DARK   = colors.HexColor("#1a1a2e")
GM_BLUE   = colors.HexColor("#16213e")
GM_ACCENT = colors.HexColor("#0f3460")
RED_COLOR   = colors.HexColor("#c0392b")
YELLOW_COLOR = colors.HexColor("#e67e22")
GREEN_COLOR  = colors.HexColor("#27ae60")
LIGHT_GRAY  = colors.HexColor("#f5f5f5")
MID_GRAY    = colors.HexColor("#cccccc")
DARK_TEXT   = colors.HexColor("#222222")

PAGE_W, PAGE_H = letter
MARGIN = 0.75 * inch


# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------
def _styles():
    base = getSampleStyleSheet()
    s = {}

    def add(name, **kw):
        s[name] = ParagraphStyle(name, **kw)

    add("cover_title",    fontName="Helvetica-Bold", fontSize=22, textColor=colors.white,
        leading=28, alignment=TA_CENTER)
    add("cover_sub",      fontName="Helvetica", fontSize=11, textColor=colors.HexColor("#aaaacc"),
        leading=16, alignment=TA_CENTER)
    add("cover_meta",     fontName="Helvetica", fontSize=9, textColor=colors.HexColor("#888888"),
        leading=13, alignment=TA_CENTER)
    add("section_header", fontName="Helvetica-Bold", fontSize=13, textColor=GM_ACCENT,
        leading=18, spaceAfter=6)
    add("body",           fontName="Helvetica", fontSize=9, textColor=DARK_TEXT,
        leading=14, spaceAfter=4)
    add("body_bold",      fontName="Helvetica-Bold", fontSize=9, textColor=DARK_TEXT,
        leading=14, spaceAfter=4)
    add("small",          fontName="Helvetica", fontSize=8, textColor=colors.HexColor("#555555"),
        leading=12, spaceAfter=2)
    add("caption",        fontName="Helvetica-Oblique", fontSize=8,
        textColor=colors.HexColor("#555555"), leading=11, spaceAfter=8)
    add("alert_text",     fontName="Helvetica-Bold", fontSize=9, textColor=colors.white,
        leading=14)
    add("risk_body",      fontName="Helvetica", fontSize=9, textColor=colors.white, leading=14)
    add("footer",         fontName="Helvetica", fontSize=7, textColor=colors.HexColor("#888888"),
        leading=10, alignment=TA_CENTER)
    add("toc_item",       fontName="Helvetica", fontSize=9, textColor=DARK_TEXT, leading=14)
    add("about",          fontName="Helvetica", fontSize=8, textColor=colors.HexColor("#444444"),
        leading=12)
    return s


# ---------------------------------------------------------------------------
# Header / footer
# ---------------------------------------------------------------------------
def _header_footer(canvas, doc):
    canvas.saveState()
    page = canvas.getPageNumber()
    if page == 1:
        canvas.restoreState()
        return
    # Header bar
    canvas.setFillColor(GM_DARK)
    canvas.rect(0, PAGE_H - 0.45 * inch, PAGE_W, 0.45 * inch, fill=1, stroke=0)
    canvas.setFillColor(colors.white)
    canvas.setFont("Helvetica-Bold", 9)
    canvas.drawString(MARGIN, PAGE_H - 0.28 * inch, "GAMMAMETRIC")
    canvas.setFont("Helvetica", 8)
    canvas.drawCentredString(PAGE_W / 2, PAGE_H - 0.28 * inch,
                             "Model Reliability Under Real-World Acquisition Conditions")
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#aaaaaa"))
    canvas.drawRightString(PAGE_W - MARGIN, PAGE_H - 0.28 * inch,
                           f"CONFIDENTIAL \u2014 GammaMetric LLC  |  gammametric.com")
    # Footer
    canvas.setFillColor(colors.HexColor("#888888"))
    canvas.setFont("Helvetica", 7)
    canvas.drawCentredString(PAGE_W / 2, 0.35 * inch, f"Page {page}")
    canvas.restoreState()


# ---------------------------------------------------------------------------
# Table helpers
# ---------------------------------------------------------------------------
def _base_table_style(row_count: int, header_bg=GM_ACCENT) -> list:
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), header_bg),
        ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",   (0, 0), (-1, -1), 8),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [LIGHT_GRAY, colors.white]),
        ("GRID",       (0, 0), (-1, -1), 0.5, MID_GRAY),
        ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ]
    return style


def _status_color(status: str) -> colors.Color:
    return RED_COLOR if "OUT" in status else GREEN_COLOR


# ---------------------------------------------------------------------------
# Range checks for site acquisition profile
# ---------------------------------------------------------------------------
VALIDATED_KERNEL_CLASSES = {"STANDARD"}
CTDIVOL_VALIDATED = (6.0, 12.0)
KVP_VALIDATED     = (100, 140)


def _kernel_status(kernel: str) -> tuple[str, str]:
    cls, _, ood = normalize_kernel(kernel)
    if ood or cls not in VALIDATED_KERNEL_CLASSES:
        return f"{kernel} ({cls.lower()})", "OUT OF RANGE"
    return f"{kernel} ({cls.lower()})", "IN RANGE"


def _slice_status(mm: float) -> str:
    return "OUT OF RANGE" if mm > 2.5 else "IN RANGE"


def _ctdi_status(mgy: float) -> str:
    lo, hi = CTDIVOL_VALIDATED
    return "OUT OF RANGE" if not (lo <= mgy <= hi) else "IN RANGE"


def _kvp_status(kvp: Optional[float]) -> str:
    if kvp is None:
        return "N/A"
    lo, hi = KVP_VALIDATED
    return "OUT OF RANGE" if not (lo <= kvp <= hi) else "IN RANGE"


# ---------------------------------------------------------------------------
# Page builders
# ---------------------------------------------------------------------------

def _page1_cover(story, s, inp: SensitivityInput, result: SensitivityResult, report_date: str):
    """Cover page with dark background."""
    # Simulate dark background with a large colored rectangle via a Table trick
    bg_data = [[""]]
    bg_table = Table(bg_data, colWidths=[PAGE_W - 2 * MARGIN], rowHeights=[1.2 * inch])
    bg_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), GM_DARK),
    ]))

    story.append(Spacer(1, 1.5 * inch))

    # Logo / company name
    story.append(Paragraph("GAMMAMETRIC", ParagraphStyle(
        "logo", fontName="Helvetica-Bold", fontSize=28,
        textColor=GM_ACCENT, alignment=TA_CENTER, spaceAfter=4)))
    story.append(Paragraph("Clinical AI Reliability Infrastructure", ParagraphStyle(
        "logo_sub", fontName="Helvetica", fontSize=12,
        textColor=colors.HexColor("#666666"), alignment=TA_CENTER, spaceAfter=0.5 * inch)))

    # Title box
    title_data = [[
        Paragraph("Model Reliability Under Real-World\nAcquisition Conditions", ParagraphStyle(
            "tb", fontName="Helvetica-Bold", fontSize=20,
            textColor=colors.white, alignment=TA_CENTER, leading=26))
    ]]
    title_table = Table(title_data, colWidths=[PAGE_W - 2 * MARGIN - 0.5 * inch])
    title_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), GM_DARK),
        ("TOPPADDING", (0, 0), (-1, -1), 24),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 24),
        ("LEFTPADDING", (0, 0), (-1, -1), 20),
        ("RIGHTPADDING", (0, 0), (-1, -1), 20),
    ]))
    story.append(title_table)
    story.append(Spacer(1, 0.15 * inch))

    # Subtitle strip
    sub_data = [[
        Paragraph(f"CT Lung Nodule Detection  |  DEMONSTRATION REPORT  |  {report_date}", ParagraphStyle(
            "sub_strip", fontName="Helvetica", fontSize=10,
            textColor=colors.HexColor("#aaaaaa"), alignment=TA_CENTER))
    ]]
    sub_table = Table(sub_data, colWidths=[PAGE_W - 2 * MARGIN - 0.5 * inch])
    sub_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), GM_BLUE),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
    ]))
    story.append(sub_table)
    story.append(Spacer(1, 0.5 * inch))

    # Metadata table
    meta = [
        ["Report Type", "Demonstration \u2014 Acquisition Robustness Characterization"],
        ["AI Task",     "CT Lung Nodule Detection (3\u201310 mm)"],
        ["Model",       "MONAI RetinaNet (RetinaNet architecture, LUNA16-trained)"],
        ["Dataset",     "154 LIDC-IDRI cases across 6 acquisition conditions"],
        ["Parameters",  "Slice thickness, Reconstruction kernel, Dose level"],
        ["Date",        report_date],
        ["Prepared by", "Daniel Soliman, M.S. \u2014 GammaMetric LLC"],
        ["Status",      "DEMONSTRATION \u2014 based on published research findings"],
    ]
    meta_table = Table(
        [[Paragraph(k, s["body_bold"]), Paragraph(v, s["body"])] for k, v in meta],
        colWidths=[1.5 * inch, PAGE_W - 2 * MARGIN - 1.5 * inch - 0.5 * inch],
    )
    meta_table.setStyle(TableStyle([
        ("FONTSIZE",  (0, 0), (-1, -1), 9),
        ("VALIGN",    (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LINEBELOW", (0, 0), (-1, -2), 0.5, MID_GRAY),
        ("LEFTPADDING", (1, 0), (1, -1), 12),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 0.4 * inch))
    story.append(HRFlowable(width="100%", thickness=1, color=MID_GRAY))
    story.append(Spacer(1, 0.15 * inch))
    story.append(Paragraph(
        "This report characterizes how a commercially representative lung nodule detection AI model performs "
        "across the range of acquisition conditions encountered in real-world clinical deployment. Performance "
        "metrics are derived from systematic perturbation experiments conducted on a 154-case LIDC-IDRI dataset. "
        "Findings are published on arXiv (2603.26785) and currently under review at <i>Academic Radiology</i>.",
        s["body"]))
    story.append(Spacer(1, 0.1 * inch))
    story.append(Paragraph(
        "<b>Key finding:</b> Under commonly deployed acquisition conditions \u2014 specifically soft reconstruction "
        "kernels and 5 mm slice thickness \u2014 sensitivity for clinically actionable 3\u20136 mm nodules drops "
        "by 10\u201313 percentage points relative to validated baseline conditions. The model\u2019s confidence "
        "score does not reflect this degradation.",
        s["body"]))
    story.append(PageBreak())


def _page2_site_profile(story, s, inp: SensitivityInput, result: SensitivityResult):
    story.append(Paragraph("1. Site Acquisition Profile", s["section_header"]))
    story.append(Paragraph(
        "The table below represents a typical deployment site\u2019s scanner acquisition parameters extracted "
        "from DICOM metadata, compared against the validated operating envelope for this model class. "
        "In a live deployment, this section is auto-populated from incoming DICOM headers.",
        s["body"]))
    story.append(Spacer(1, 0.15 * inch))

    kernel_display, kernel_status = _kernel_status(inp.reconstruction_kernel)
    slice_status  = _slice_status(inp.slice_thickness_mm)
    ctdi_status   = _ctdi_status(inp.ctdivol_mgy)
    scanner_status = "IN RANGE"

    rows = [
        ["Parameter", "Site Value", "Validated Range", "Status"],
        ["Slice Thickness",       f"{inp.slice_thickness_mm} mm",   "\u2264 2.5 mm",       slice_status],
        ["Reconstruction Kernel", kernel_display,                    "B30f\u2013B35f",      kernel_status],
        ["CTDIvol",               f"{inp.ctdivol_mgy} mGy",         "6.0\u201312.0 mGy",   ctdi_status],
        ["Scanner Model",         inp.scanner_model or "Not specified", "Multi-vendor",    scanner_status],
        ["Tube Voltage",          "120 kVp",                         "100\u2013140 kVp",    "IN RANGE"],
        ["Reconstruction Method", "Iterative / FBP",                 "FBP or IR",           "IN RANGE"],
    ]

    col_w = [2.2 * inch, 1.6 * inch, 1.6 * inch, 1.2 * inch]
    table_data = []
    for i, row in enumerate(rows):
        if i == 0:
            table_data.append([Paragraph(c, ParagraphStyle(
                "th", fontName="Helvetica-Bold", fontSize=8, textColor=colors.white)) for c in row])
        else:
            status = row[3]
            status_color = RED_COLOR if "OUT" in status else GREEN_COLOR
            table_data.append([
                Paragraph(row[0], s["body"]),
                Paragraph(row[1], s["body"]),
                Paragraph(row[2], s["body"]),
                Paragraph(f'<font color="{"#c0392b" if "OUT" in status else "#27ae60"}"><b>{status}</b></font>',
                          s["body"]),
            ])

    t = Table(table_data, colWidths=col_w)
    ts = _base_table_style(len(rows))
    t.setStyle(TableStyle(ts))
    story.append(t)
    story.append(Spacer(1, 0.1 * inch))
    story.append(Paragraph(
        "Figure 1. Acquisition parameter compliance summary. Parameters shown in red fall outside the "
        "validated operating envelope.", s["caption"]))

    # Alert box
    out_of_range = sum(1 for r in rows[1:] if "OUT" in r[3])
    alert_text = (
        f"\u25a0 RELIABILITY ALERT: This site is operating the AI model outside its validated acquisition "
        f"conditions on {out_of_range} key {'parameter' if out_of_range == 1 else 'parameters'}. "
        f"Based on characterized sensitivity curves, estimated performance degradation for 3\u20136 mm nodules "
        f"is {result.degradation_pp:.0f}\u201324 percentage points below validated baseline. "
        f"The model\u2019s confidence scores do not reflect this degradation."
    )
    alert_data = [[Paragraph(alert_text, s["alert_text"])]]
    alert_table = Table(alert_data, colWidths=[PAGE_W - 2 * MARGIN - 0.5 * inch])
    alert_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), RED_COLOR),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
    ]))
    story.append(Spacer(1, 0.15 * inch))
    story.append(alert_table)
    story.append(PageBreak())


def _page3_baseline(story, s):
    story.append(Paragraph("2. Baseline Performance Summary", s["section_header"]))
    story.append(Paragraph(
        "Baseline performance reflects model behavior under validated acquisition conditions: 1.25 mm slice "
        "thickness, standard (B30f) reconstruction kernel, and full-dose imaging. These conditions represent "
        "the parameters used in model validation and FDA clearance submissions.",
        s["body"]))
    story.append(Spacer(1, 0.15 * inch))

    # Key metrics boxes
    metrics = [
        ("84.8%", "Overall Sensitivity", "All nodule sizes"),
        ("78.2%", "Sensitivity 3\u20136 mm", "Clinically critical range"),
        ("91.3%", "Sensitivity 6\u201310 mm", "Larger nodules"),
        ("0.8/scan", "False Positive Rate", "Per scan average"),
    ]
    box_data = []
    for val, label, sub in metrics:
        box_data.append(
            Paragraph(f'<font size="18"><b>{val}</b></font><br/>'
                      f'<font size="9"><b>{label}</b></font><br/>'
                      f'<font size="7" color="#666666">{sub}</font>',
                      ParagraphStyle("metric", alignment=TA_CENTER, leading=16))
        )
    metric_table = Table([box_data], colWidths=[(PAGE_W - 2 * MARGIN - 0.5 * inch) / 4] * 4)
    metric_table.setStyle(TableStyle([
        ("BOX",        (0, 0), (-1, -1), 0.5, MID_GRAY),
        ("INNERGRID",  (0, 0), (-1, -1), 0.5, MID_GRAY),
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT_GRAY),
        ("BACKGROUND", (1, 0), (1, 0),   colors.HexColor("#eaf4ea")),
        ("TOPPADDING", (0, 0), (-1, -1), 12),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
        ("ALIGN",      (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(metric_table)
    story.append(Spacer(1, 0.15 * inch))
    story.append(Paragraph(
        "The 3\u20136 mm nodule range is the clinically critical window where detection decisions have the "
        "greatest downstream consequence. Nodules in this range require active Fleischner follow-up; missed "
        "detections at 4 mm can present as 8+ mm lesions two years later. Baseline sensitivity of 78.2% in "
        "this range establishes the performance ceiling against which acquisition-driven degradation is measured.",
        s["body"]))
    story.append(Spacer(1, 0.15 * inch))

    rows = [
        ["Nodule Size", "Sensitivity", "Cases (n)", "95% CI", "Clinical Relevance"],
        ["< 3 mm",  "41.2%", "34", "(28.7\u201354.8%)", "Below Fleischner threshold (low-risk)"],
        ["3\u20136 mm",  "78.2%", "67", "(67.1\u201387.1%)", "Active follow-up required \u2014 CRITICAL"],
        ["6\u201310 mm", "91.3%", "38", "(78.2\u201397.6%)", "High detection confidence"],
        ["> 10 mm", "97.1%", "15", "(83.8\u201399.9%)", "Near-certain detection"],
    ]
    col_w = [0.9 * inch, 0.9 * inch, 0.8 * inch, 1.2 * inch, 2.8 * inch]
    t_data = []
    for i, row in enumerate(rows):
        style_name = "body_bold" if i == 0 else "body"
        t_data.append([Paragraph(c, s[style_name]) for c in row])
    t = Table(t_data, colWidths=col_w)
    ts = _base_table_style(len(rows))
    ts.append(("BACKGROUND", (0, 2), (-1, 2), colors.HexColor("#ffeaea")))  # highlight 3-6mm row
    t.setStyle(TableStyle(ts))
    story.append(t)
    story.append(Spacer(1, 0.08 * inch))
    story.append(Paragraph(
        "Table 1. Size-stratified sensitivity at baseline (1.25 mm, B30f kernel, full dose). "
        "The 3\u20136 mm row (highlighted) represents the clinically critical detection window.",
        s["caption"]))
    story.append(PageBreak())


def _page4_degradation(story, s):
    story.append(Paragraph("3. Acquisition Parameter Degradation Analysis", s["section_header"]))
    story.append(Paragraph(
        "The following analysis characterizes sensitivity changes as individual acquisition parameters "
        "deviate from validated baseline conditions. Each parameter was systematically varied while holding "
        "others constant. McNemar statistical tests and bootstrap confidence intervals (1,000 iterations) "
        "were used to assess significance.",
        s["body"]))
    story.append(Spacer(1, 0.12 * inch))

    # 3a Slice thickness
    story.append(Paragraph("3a. Slice Thickness", s["body_bold"]))
    slice_rows = [
        ["Slice Thickness", "Overall Sensitivity", "3\u20136 mm Sensitivity", "\u0394 vs Baseline", "p-value"],
        ["1.25 mm (baseline)", "84.8%", "78.2%", "\u2014", "\u2014"],
        ["2.5 mm",  "82.1%", "75.4%", "\u22122.7 pp", "0.31"],
        ["3.75 mm", "78.3%", "70.1%", "\u22128.1 pp", "0.04"],
        ["5.0 mm",  "71.6%", "65.0%", "\u221213.2 pp", "<0.001"],
    ]
    col_w = [1.4*inch, 1.3*inch, 1.4*inch, 1.1*inch, 0.9*inch]
    t_data = [[Paragraph(c, s["body_bold"] if i == 0 else s["body"]) for c in row]
              for i, row in enumerate(slice_rows)]
    t = Table(t_data, colWidths=col_w)
    ts = _base_table_style(len(slice_rows))
    ts.append(("BACKGROUND", (0, 4), (-1, 4), colors.HexColor("#ffeaea")))
    t.setStyle(TableStyle(ts))
    story.append(t)
    story.append(Spacer(1, 0.06 * inch))
    story.append(Paragraph(
        "Table 2. Sensitivity by slice thickness. The 5 mm condition (highlighted red) produces a "
        "statistically significant 13.2 pp overall sensitivity reduction.", s["caption"]))
    story.append(Spacer(1, 0.12 * inch))

    # 3b Kernel
    story.append(Paragraph("3b. Reconstruction Kernel", s["body_bold"]))
    kernel_rows = [
        ["Kernel", "Type", "Overall Sensitivity", "3\u20136 mm Sensitivity", "\u0394 vs Baseline"],
        ["B30f (baseline)", "Standard", "84.8%", "78.2%", "\u2014"],
        ["B35f",  "Standard", "83.9%", "77.1%", "\u22121.1 pp"],
        ["B40f",  "Soft",     "76.8%", "70.3%", "\u22127.9 pp"],
        ["B50f",  "Soft/smooth", "74.3%", "67.7%", "\u221210.5 pp"],
    ]
    col_w = [1.3*inch, 1.0*inch, 1.3*inch, 1.4*inch, 1.1*inch]
    t_data = [[Paragraph(c, s["body_bold"] if i == 0 else s["body"]) for c in row]
              for i, row in enumerate(kernel_rows)]
    t = Table(t_data, colWidths=col_w)
    ts = _base_table_style(len(kernel_rows))
    ts.append(("BACKGROUND", (0, 3), (-1, 3), colors.HexColor("#fff3e0")))
    ts.append(("BACKGROUND", (0, 4), (-1, 4), colors.HexColor("#ffeaea")))
    t.setStyle(TableStyle(ts))
    story.append(t)
    story.append(Spacer(1, 0.06 * inch))
    story.append(Paragraph(
        "Table 3. Sensitivity by reconstruction kernel. Soft kernels (B40f, B50f) produce the largest "
        "sensitivity reductions, disproportionately affecting the 3\u20136 mm range.", s["caption"]))
    story.append(Spacer(1, 0.12 * inch))

    # 3c Dose
    story.append(Paragraph("3c. Dose Level", s["body_bold"]))
    dose_rows = [
        ["Dose Level", "CTDIvol", "Overall Sensitivity", "3\u20136 mm Sensitivity", "\u0394 vs Baseline"],
        ["Full dose (baseline)", "~10 mGy", "84.8%", "78.2%", "\u2014"],
        ["75% dose",           "~7.5 mGy", "83.6%", "77.0%", "\u22121.2 pp"],
        ["50% dose (LDCT)",    "~5 mGy",   "82.1%", "75.9%", "\u22122.3 pp"],
        ["25% dose (ultra-low)", "~2.5 mGy", "79.4%", "73.1%", "\u22125.1 pp"],
    ]
    col_w = [1.5*inch, 0.9*inch, 1.3*inch, 1.4*inch, 1.1*inch]
    t_data = [[Paragraph(c, s["body_bold"] if i == 0 else s["body"]) for c in row]
              for i, row in enumerate(dose_rows)]
    t = Table(t_data, colWidths=col_w)
    t.setStyle(TableStyle(_base_table_style(len(dose_rows))))
    story.append(t)
    story.append(Spacer(1, 0.06 * inch))
    story.append(Paragraph(
        "Table 4. Sensitivity by dose level. Dose reduction produces modest, statistically non-significant "
        "sensitivity changes compared to kernel and slice thickness.", s["caption"]))
    story.append(PageBreak())


def _page5_combined(story, s, inp: SensitivityInput, result: SensitivityResult):
    story.append(Paragraph(
        "4. Combined Parameter Effects & Site-Specific Risk Assessment", s["section_header"]))

    # Identify which drivers are active
    driver_map = {d["parameter"]: d for d in result.drivers}
    has_slice  = "slice_thickness" in driver_map
    has_kernel = "kernel" in driver_map

    story.append(Paragraph(
        "When multiple parameters deviate simultaneously from validated conditions, effects compound. "
        "The site profile in Section 1 shows out-of-range parameters and their combined effect on "
        "AI sensitivity. The combined sensitivity estimate is derived from the characterized sensitivity functions.",
        s["body"]))
    story.append(Spacer(1, 0.12 * inch))

    est_sensitivity_pct = f"{result.estimated_sensitivity:.1%}"
    degradation = result.degradation_pp
    baseline_36 = 78.2

    rows = [["Condition", "Overall Sensitivity", "3\u20136 mm Sensitivity", "Cumulative \u0394"]]
    rows.append(["Baseline (1.25 mm, B30f, full dose)", "84.8%", "78.2%", "\u2014"])

    cumulative = 0.0
    if has_slice:
        delta = driver_map["slice_thickness"]["contribution_pp"]
        cumulative += delta
        rows.append([
            f"+ {inp.slice_thickness_mm} mm slice thickness only",
            f"{84.8 + cumulative:.1f}%",
            f"{baseline_36 + cumulative:.1f}%",
            f"{cumulative:.1f} pp",
        ])

    if has_kernel:
        delta = driver_map["kernel"]["contribution_pp"]
        cumulative += delta
        rows.append([
            f"+ {inp.reconstruction_kernel} kernel only",
            f"{84.8 + result.drivers[-1]['contribution_pp'] + (driver_map['slice_thickness']['contribution_pp'] if has_slice else 0):.1f}%",
            f"{baseline_36 + (driver_map['slice_thickness']['contribution_pp'] if has_slice else 0) + delta:.1f}%",
            f"{(driver_map['slice_thickness']['contribution_pp'] if has_slice else 0) + delta:.1f} pp",
        ])

    rows.append([
        "Both (this site\u2019s conditions)",
        f"{84.8 - degradation:.1f}%",
        est_sensitivity_pct,
        f"\u2212{degradation:.1f} pp",
    ])

    col_w = [2.5*inch, 1.3*inch, 1.4*inch, 1.0*inch]
    t_data = [[Paragraph(c, s["body_bold"] if i == 0 else s["body"]) for c in row]
              for i, row in enumerate(rows)]
    t = Table(t_data, colWidths=col_w)
    ts = _base_table_style(len(rows))
    ts.append(("BACKGROUND", (0, len(rows)-1), (-1, len(rows)-1), colors.HexColor("#ffeaea")))
    t.setStyle(TableStyle(ts))
    story.append(t)
    story.append(Spacer(1, 0.06 * inch))
    story.append(Paragraph(
        f"Table 5. Combined parameter effects. Under this site\u2019s current acquisition conditions, "
        f"3\u20136 mm nodule sensitivity is estimated at {est_sensitivity_pct} \u2014 a {degradation:.1f} pp "
        f"reduction from validated baseline. Approximately 1 in "
        f"{round(1/(degradation/100)):.0f} clinically actionable small nodules may be missed.",
        s["caption"]))
    story.append(Spacer(1, 0.2 * inch))

    # Risk summary box
    classification_color = {"RED": RED_COLOR, "YELLOW": YELLOW_COLOR, "GREEN": GREEN_COLOR}[result.classification]
    risk_text = (
        f"Under current acquisition conditions, this deployment site is estimated to be operating at "
        f"{est_sensitivity_pct} sensitivity for 3\u20136 mm nodules \u2014 the range requiring active "
        f"Fleischner follow-up. This represents a {degradation:.1f} percentage point reduction from the "
        f"validated baseline of 78.2%.\n\n"
        f"The AI model\u2019s confidence scores do not reflect this degradation. Radiologists relying on "
        f"the model\u2019s output at this site are receiving signals calibrated to validated conditions "
        f"that do not currently exist."
    )
    risk_data = [
        [Paragraph("SITE RISK SUMMARY", ParagraphStyle(
            "risk_hdr", fontName="Helvetica-Bold", fontSize=10, textColor=colors.white))],
        [Paragraph(risk_text, s["risk_body"])],
    ]
    risk_table = Table(risk_data, colWidths=[PAGE_W - 2 * MARGIN - 0.5 * inch])
    risk_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), classification_color),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("LEFTPADDING", (0, 0), (-1, -1), 14),
        ("RIGHTPADDING", (0, 0), (-1, -1), 14),
    ]))
    story.append(risk_table)
    story.append(PageBreak())


def _page6_recommendations(story, s, inp: SensitivityInput, result: SensitivityResult):
    story.append(Paragraph("5. Recommendations", s["section_header"]))
    story.append(Paragraph(
        "The following protocol adjustments are recommended to restore AI model performance to within "
        "the validated operating envelope. Recommendations are prioritized by estimated sensitivity recovery.",
        s["body"]))
    story.append(Spacer(1, 0.12 * inch))

    driver_map = {d["parameter"]: d for d in result.drivers}
    recs = []

    if "slice_thickness" in driver_map:
        recs.append([
            "1 \u2014 HIGH",
            "Reduce slice thickness reconstruction to \u2264 2.5 mm for LDCT lung screening series. "
            "Retain thicker slices for other reads if needed but ensure thin-slice series is always "
            "available for AI processing.",
            "+13.2 pp overall;\n+13.2 pp for 3\u20136 mm range",
            "High \u2014 software/protocol change only",
        ])
    if "kernel" in driver_map:
        recs.append([
            f"{'2' if len(recs) else '1'} \u2014 HIGH",
            f"Switch reconstruction kernel from {inp.reconstruction_kernel} to B30f or B35f for lung "
            f"nodule AI input series. Soft kernel can be retained for other clinical uses; AI input "
            f"should be routed from the standard kernel reconstruction.",
            f"+{abs(driver_map['kernel']['contribution_pp']):.1f} pp overall;\n"
            f"+{abs(driver_map['kernel']['contribution_pp']):.1f} pp for 3\u20136 mm range",
            "High \u2014 dual reconstruction if needed",
        ])

    recs.append([
        f"{len(recs)+1} \u2014 MEDIUM",
        "Implement GammaMetric continuous acquisition monitoring to alert when parameters drift outside "
        "validated envelope in real time. Ensures corrective action before extended degraded operation.",
        "Prevents future drift",
        "Medium \u2014 requires DICOM header monitoring",
    ])
    recs.append([
        f"{len(recs)+1} \u2014 LOW",
        "Document current acquisition parameters against AI vendor IFU validated specifications. "
        "Ensure any future scanner upgrades or protocol changes trigger re-validation review.",
        "Compliance / liability",
        "Low \u2014 administrative",
    ])

    col_w = [0.85*inch, 2.6*inch, 1.4*inch, 1.35*inch]
    headers = ["Priority", "Action", "Expected Sensitivity Recovery", "Feasibility"]
    t_data = [[Paragraph(h, s["body_bold"]) for h in headers]]
    for rec in recs:
        t_data.append([Paragraph(c, s["body"]) for c in rec])
    t = Table(t_data, colWidths=col_w)
    ts = _base_table_style(len(t_data))
    t.setStyle(TableStyle(ts))
    story.append(t)
    story.append(Spacer(1, 0.15 * inch))

    if "slice_thickness" in driver_map and "kernel" in driver_map:
        story.append(Paragraph(
            f"Implementation of Recommendations 1 and 2 together is estimated to restore 3\u20136 mm nodule "
            f"sensitivity from {result.estimated_sensitivity:.1%} to approximately 78.2% \u2014 returning "
            f"performance to the validated baseline. Both changes are achievable through scanner protocol "
            f"modification without hardware upgrades.",
            s["body"]))
    story.append(PageBreak())


def _page7_methodology(story, s):
    story.append(Paragraph("6. Methodology & Credentialing Summary", s["section_header"]))
    story.append(Paragraph(
        "This report is based on systematic experimental characterization, not statistical proxy measures "
        "or distribution shift detection. Performance estimates are derived from controlled perturbation "
        "experiments with ground-truth labels, enabling quantitative sensitivity predictions rather than "
        "anomaly flags.",
        s["body"]))
    story.append(Spacer(1, 0.12 * inch))

    method_rows = [
        ["Dataset", "154 LIDC-IDRI cases with expert consensus nodule annotations. Cases selected to "
         "represent the 3\u201310 mm nodule range with balanced size distribution."],
        ["Model", "MONAI RetinaNet trained on LUNA16. Representative of the RetinaNet-class architectures "
         "used in multiple FDA-cleared lung nodule detection tools."],
        ["Perturbation Method", "Each case processed under 6 acquisition conditions via retrospective "
         "reconstruction simulation. Conditions held one parameter variable while others remained at baseline."],
        ["Statistical Methods", "McNemar\u2019s test for paired sensitivity comparisons. Bootstrap confidence "
         "intervals (1,000 iterations). FROC curve analysis for full operating characteristic characterization."],
        ["Validation", "Findings submitted to Academic Radiology (under review). Preprint available: "
         "arXiv:2603.26785 (cs.CV)."],
        ["Limitations", "Characterization currently covers one model architecture on one dataset. Expansion "
         "to additional models and modalities requires additional experimental work. Retrospective reconstruction "
         "simulation is an approximation of true scanner-level parameter variation."],
    ]
    col_w = [1.4 * inch, PAGE_W - 2 * MARGIN - 1.4 * inch - 0.5 * inch]
    t_data = [[Paragraph(k, s["body_bold"]), Paragraph(v, s["body"])] for k, v in method_rows]
    t = Table(t_data, colWidths=col_w)
    t.setStyle(TableStyle([
        ("FONTSIZE",   (0, 0), (-1, -1), 9),
        ("VALIGN",     (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LINEBELOW",  (0, 0), (-1, -2), 0.5, MID_GRAY),
        ("LEFTPADDING", (1, 0), (1, -1), 12),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [LIGHT_GRAY, colors.white]),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.2 * inch))

    # Why physics-grounded box
    why_text = (
        "<b>Why physics-grounded characterization matters:</b> Most post-deployment monitoring systems detect "
        "that <i>something</i> has changed \u2014 they flag statistical divergence from a reference distribution. "
        "GammaMetric\u2019s approach characterizes <i>how much</i> performance has degraded and in which "
        "detection range, using experimentally derived sensitivity functions rather than proxy metrics. "
        "This enables quantitative risk assessment, not just anomaly detection."
    )
    why_data = [[Paragraph(why_text, ParagraphStyle(
        "why", fontName="Helvetica", fontSize=9, textColor=DARK_TEXT, leading=14))]]
    why_table = Table(why_data, colWidths=[PAGE_W - 2 * MARGIN - 0.5 * inch])
    why_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), colors.HexColor("#e8f0fe")),
        ("TOPPADDING",    (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("LEFTPADDING",   (0, 0), (-1, -1), 14),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 14),
    ]))
    story.append(why_table)
    story.append(Spacer(1, 0.25 * inch))
    story.append(HRFlowable(width="100%", thickness=0.5, color=MID_GRAY))
    story.append(Spacer(1, 0.1 * inch))

    about = (
        "<b>About GammaMetric</b><br/>"
        "GammaMetric LLC provides clinical AI reliability infrastructure for medical imaging. Our "
        "physics-grounded reliability engine characterizes how imaging AI performance changes across "
        "real-world acquisition conditions, enabling AI vendors and health systems to understand, monitor, "
        "and document deployed model performance. Founded by Daniel Soliman, M.S. \u2014 Diagnostic Medical "
        "Physicist (Yale residency, ABR board-eligible).<br/><br/>"
        "daniel@gammametric.com \u2014 gammametric.com \u2014 arXiv:2603.26785"
    )
    story.append(Paragraph(about, s["about"]))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_report(inp: SensitivityInput, result: Optional[SensitivityResult] = None) -> bytes:
    if result is None:
        result = compute(inp)

    buf = BytesIO()
    s = _styles()
    report_date = datetime.utcnow().strftime("%B %Y")

    doc = BaseDocTemplate(
        buf,
        pagesize=letter,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=0.55 * inch,
        bottomMargin=0.6 * inch,
    )
    frame = Frame(MARGIN, 0.6 * inch, PAGE_W - 2 * MARGIN, PAGE_H - 1.15 * inch, id="main")
    cover_frame = Frame(MARGIN, 0.6 * inch, PAGE_W - 2 * MARGIN, PAGE_H - 1.2 * inch, id="cover")
    doc.addPageTemplates([
        PageTemplate(id="cover", frames=[cover_frame]),
        PageTemplate(id="body",  frames=[frame], onPage=_header_footer),
    ])

    story = []
    _page1_cover(story, s, inp, result, report_date)
    story.append(PageBreak())
    _page2_site_profile(story, s, inp, result)
    _page3_baseline(story, s)
    _page4_degradation(story, s)
    _page5_combined(story, s, inp, result)
    _page6_recommendations(story, s, inp, result)
    _page7_methodology(story, s)

    doc.build(story)
    return buf.getvalue()
