"""
LeapfrogDose Web App
====================
FastAPI app — upload a Radimetrics CSV, get a Leapfrog-ready HTML report.

Email delivery: set SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS env vars to enable.
"""

import os
import sys
import smtplib
import tempfile
import contextlib
import io
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

sys.path.insert(0, str(Path(__file__).parent.parent))
from core.leapfrog_dose import load_radimetrics_csv, analyze_facility, COLUMN_MAPPINGS, BENCHMARK_VERSION

SMTP_HOST = os.getenv("SMTP_HOST", "")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASS = os.getenv("SMTP_PASS", "")
EMAIL_FROM = os.getenv("EMAIL_FROM", SMTP_USER)
EMAIL_ENABLED = bool(SMTP_HOST and SMTP_USER and SMTP_PASS)

app = FastAPI(title="LeapfrogDose", docs_url=None, redoc_url=None)
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")


def _column_report(df) -> dict:
    """
    Return which expected fields were found vs. missing after CSV load.
    Groups into critical (DLP, body region) and optional.
    """
    CRITICAL = ["dlp", "study_description", "scan_region", "age"]
    OPTIONAL  = ["exam_date", "scanner", "patient_id", "ctdivol", "ssde"]

    found    = [f for f in CRITICAL + OPTIONAL if f in df.columns]
    missing  = [f for f in CRITICAL if f not in df.columns]
    optional_missing = [f for f in OPTIONAL if f not in df.columns]

    # At least one of study_description / scan_region needed for body region
    region_ok = "study_description" in df.columns or "scan_region" in df.columns

    return {
        "found": found,
        "critical_missing": missing,
        "optional_missing": optional_missing,
        "region_ok": region_ok,
        "dlp_ok": "dlp" in df.columns,
        "age_ok": "age" in df.columns,
    }


def _send_report_email(to_address: str, facility_name: str, html: str):
    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"LeapfrogDose Report — {facility_name}"
    msg["From"]    = f"LeapfrogDose <{EMAIL_FROM}>"
    msg["To"]      = to_address

    msg.attach(MIMEText(
        f"Your Leapfrog Section 8B dose report for {facility_name} is attached below.\n\n"
        f"— LeapfrogDose by GammaMetric",
        "plain"
    ))
    msg.attach(MIMEText(html, "html"))

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.ehlo()
        server.starttls()
        server.login(SMTP_USER, SMTP_PASS)
        server.sendmail(EMAIL_FROM, to_address, msg.as_string())


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "email_enabled": EMAIL_ENABLED},
    )


@app.post("/analyze", response_class=HTMLResponse)
async def analyze(
    request: Request,
    file: UploadFile = File(...),
    facility_name: str = Form(default="Facility"),
    email: str = Form(default=""),
):
    # ── Validate file type ──────────────────────────────────────────
    if not file.filename.lower().endswith(".csv"):
        return templates.TemplateResponse(
            "error.html",
            {
                "request": request,
                "title": "Wrong file type",
                "message": "Please upload a CSV file (.csv). "
                           "Export your dose data from Radimetrics as CSV and try again.",
                "suggestions": [
                    "In Radimetrics: Reports → Dose Tracking → Export → CSV",
                    "Ensure the file extension is .csv, not .xlsx or .xls",
                ],
            },
            status_code=400,
        )

    contents = await file.read()
    if len(contents) == 0:
        return templates.TemplateResponse(
            "error.html",
            {
                "request": request,
                "title": "Empty file",
                "message": "The uploaded file appears to be empty.",
                "suggestions": ["Check that the export completed before downloading."],
            },
            status_code=400,
        )

    # ── Parse CSV ───────────────────────────────────────────────────
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        with contextlib.redirect_stdout(io.StringIO()):
            df = load_radimetrics_csv(tmp_path)
    except Exception as e:
        Path(tmp_path).unlink(missing_ok=True)
        return templates.TemplateResponse(
            "error.html",
            {
                "request": request,
                "title": "Could not parse CSV",
                "message": str(e),
                "suggestions": [
                    "Ensure the file is a valid comma-, tab-, or semicolon-separated CSV.",
                    "Try opening in Excel and re-saving as CSV (UTF-8).",
                    "Check that the file contains a DLP column and a study description or body region column.",
                ],
            },
            status_code=422,
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    col_info = _column_report(df)

    # ── Block if no usable data ─────────────────────────────────────
    if not col_info["dlp_ok"]:
        expected = ", ".join(COLUMN_MAPPINGS["dlp"])
        return templates.TemplateResponse(
            "error.html",
            {
                "request": request,
                "title": "DLP column not found",
                "message": (
                    f"A DLP (dose-length product) column is required but was not found "
                    f"in your CSV. LeapfrogDose looked for columns named: {expected}."
                ),
                "suggestions": [
                    "Check that your Radimetrics export includes DLP values.",
                    "If your column has a different name, rename it to 'DLP' in Excel and re-upload.",
                    f"Columns found in your file: {', '.join(df.columns.tolist()[:12])}{'…' if len(df.columns) > 12 else ''}",
                ],
            },
            status_code=422,
        )

    if not col_info["region_ok"]:
        return templates.TemplateResponse(
            "error.html",
            {
                "request": request,
                "title": "No body region information found",
                "message": (
                    "LeapfrogDose needs either a Study Description or Body Region "
                    "column to classify exams. Neither was found."
                ),
                "suggestions": [
                    "Ensure your export includes the 'Study Description' or 'Body Region' column.",
                    f"Columns found in your file: {', '.join(df.columns.tolist()[:12])}{'…' if len(df.columns) > 12 else ''}",
                ],
            },
            status_code=422,
        )

    # ── Run analysis ────────────────────────────────────────────────
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            results = analyze_facility(df, facility_name.strip() or "Facility")
    except Exception as e:
        return templates.TemplateResponse(
            "error.html",
            {
                "request": request,
                "title": "Analysis failed",
                "message": str(e),
                "suggestions": [
                    "Ensure DLP values are numeric (no units in the cell).",
                    "Check that age values are numbers (years), not strings like '45y'.",
                ],
            },
            status_code=500,
        )

    # ── Render report ───────────────────────────────────────────────
    ctx = {
        "request": request,
        "results": results,
        "col_info": col_info,
        "benchmark_version": BENCHMARK_VERSION,
        "email_enabled": EMAIL_ENABLED,
        "filename": file.filename,
    }

    report_html = templates.TemplateResponse("report.html", ctx)

    # ── Email delivery (optional) ───────────────────────────────────
    email = email.strip()
    if email and EMAIL_ENABLED:
        try:
            rendered = (await report_html.body()).decode()  # type: ignore[attr-defined]
        except Exception:
            rendered = report_html.body.decode() if hasattr(report_html, "body") else ""

        try:
            _send_report_email(email, results["facility_name"], rendered)
        except Exception:
            pass  # Don't fail the response if email fails

    return report_html
