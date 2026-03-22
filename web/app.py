"""
LeapfrogDose Web App
====================
FastAPI app — upload a Radimetrics CSV, get a Leapfrog-ready HTML report.
Accounts + history storage for Phase 1 platform.

Email delivery: set SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS env vars to enable.
"""

import os
import sys
import json
import smtplib
import tempfile
import contextlib
import io
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Form, Request, Depends
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).parent.parent))
from core.leapfrog_dose import load_radimetrics_csv, analyze_facility, COLUMN_MAPPINGS, BENCHMARK_VERSION
from web.models import init_db, get_db, User, AnalysisResult
from web.auth import hash_password, verify_password, set_session, clear_session, get_user_id

# ── Email config ────────────────────────────────────────────────────────────
SMTP_HOST    = os.getenv("SMTP_HOST", "")
SMTP_PORT    = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER    = os.getenv("SMTP_USER", "")
SMTP_PASS    = os.getenv("SMTP_PASS", "")
EMAIL_FROM   = os.getenv("EMAIL_FROM", SMTP_USER)
EMAIL_ENABLED = bool(SMTP_HOST and SMTP_USER and SMTP_PASS)

# ── App setup ───────────────────────────────────────────────────────────────
app = FastAPI(title="LeapfrogDose", docs_url=None, redoc_url=None)
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")

@app.on_event("startup")
def startup():
    init_db()


# ── Helpers ─────────────────────────────────────────────────────────────────

def _column_report(df) -> dict:
    CRITICAL = ["dlp", "study_description", "scan_region", "age"]
    OPTIONAL  = ["exam_date", "scanner", "patient_id", "ctdivol", "ssde"]
    found             = [f for f in CRITICAL + OPTIONAL if f in df.columns]
    critical_missing  = [f for f in CRITICAL if f not in df.columns]
    optional_missing  = [f for f in OPTIONAL if f not in df.columns]
    return {
        "found": found,
        "critical_missing": critical_missing,
        "optional_missing": optional_missing,
        "region_ok": "study_description" in df.columns or "scan_region" in df.columns,
        "dlp_ok": "dlp" in df.columns,
        "age_ok":  "age" in df.columns,
    }


def _get_current_user(request: Request, db: Session) -> User | None:
    uid = get_user_id(request)
    if uid is None:
        return None
    return db.query(User).filter(User.id == uid).first()


def _error(request, title, message, suggestions=None, status=422):
    return templates.TemplateResponse(
        "error.html",
        {"request": request, "title": title,
         "message": message, "suggestions": suggestions or []},
        status_code=status,
    )


def _send_report_email(to_address: str, facility_name: str, html: str):
    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"LeapfrogDose Report — {facility_name}"
    msg["From"]    = f"LeapfrogDose <{EMAIL_FROM}>"
    msg["To"]      = to_address
    msg.attach(MIMEText(
        f"Your Leapfrog Section 8B dose report for {facility_name} is attached.\n\n"
        "— LeapfrogDose by GammaMetric", "plain"
    ))
    msg.attach(MIMEText(html, "html"))
    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as s:
        s.ehlo(); s.starttls(); s.login(SMTP_USER, SMTP_PASS)
        s.sendmail(EMAIL_FROM, to_address, msg.as_string())


# ── Public routes ────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request, db: Session = Depends(get_db)):
    user = _get_current_user(request, db)
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "email_enabled": EMAIL_ENABLED, "user": user},
    )


@app.get("/signup", response_class=HTMLResponse)
async def signup_get(request: Request, db: Session = Depends(get_db)):
    if _get_current_user(request, db):
        return RedirectResponse("/dashboard", status_code=302)
    return templates.TemplateResponse("signup.html", {"request": request, "error": None})


@app.post("/signup", response_class=HTMLResponse)
async def signup_post(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    facility_name: str = Form(default=""),
    db: Session = Depends(get_db),
):
    email = email.strip().lower()
    if db.query(User).filter(User.email == email).first():
        return templates.TemplateResponse(
            "signup.html",
            {"request": request, "error": "An account with that email already exists."},
        )
    if len(password) < 8:
        return templates.TemplateResponse(
            "signup.html",
            {"request": request, "error": "Password must be at least 8 characters."},
        )
    user = User(email=email, password_hash=hash_password(password),
                facility_name=facility_name.strip())
    db.add(user); db.commit(); db.refresh(user)
    response = RedirectResponse("/dashboard", status_code=302)
    set_session(response, user.id)
    return response


@app.get("/login", response_class=HTMLResponse)
async def login_get(request: Request, db: Session = Depends(get_db)):
    if _get_current_user(request, db):
        return RedirectResponse("/dashboard", status_code=302)
    return templates.TemplateResponse("login.html", {"request": request, "error": None})


@app.post("/login", response_class=HTMLResponse)
async def login_post(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db),
):
    email = email.strip().lower()
    user = db.query(User).filter(User.email == email).first()
    if not user or not verify_password(password, user.password_hash):
        return templates.TemplateResponse(
            "login.html",
            {"request": request, "error": "Incorrect email or password."},
        )
    response = RedirectResponse("/dashboard", status_code=302)
    set_session(response, user.id)
    return response


@app.get("/logout")
async def logout(request: Request):
    response = RedirectResponse("/", status_code=302)
    clear_session(response)
    return response


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request, db: Session = Depends(get_db)):
    user = _get_current_user(request, db)
    if not user:
        return RedirectResponse("/login", status_code=302)

    analyses = user.analyses  # ordered newest first

    # Build trend data: per-region P75 over time for chart
    trend = {}  # region -> [(date_str, p75), ...]
    for a in reversed(analyses):
        r = a.results
        date_str = a.analyzed_at.strftime("%Y-%m-%d")
        for region, data in r.get("adult", {}).items():
            trend.setdefault(region, []).append({
                "date": date_str,
                "p75":  data["stats"]["p75"],
                "status": data["benchmark_comparison"]["status"],
            })

    return templates.TemplateResponse("dashboard.html", {
        "request":  request,
        "user":     user,
        "analyses": analyses,
        "trend":    trend,
    })


@app.get("/report/{analysis_id}", response_class=HTMLResponse)
async def view_report(analysis_id: int, request: Request, db: Session = Depends(get_db)):
    user = _get_current_user(request, db)
    if not user:
        return RedirectResponse("/login", status_code=302)
    analysis = db.query(AnalysisResult).filter(
        AnalysisResult.id == analysis_id,
        AnalysisResult.user_id == user.id,
    ).first()
    if not analysis:
        return _error(request, "Report not found",
                      "This report doesn't exist or doesn't belong to your account.", status=404)
    return templates.TemplateResponse("report.html", {
        "request":           request,
        "results":           analysis.results,
        "col_info":          {"found": [], "critical_missing": [], "optional_missing": [],
                              "region_ok": True, "dlp_ok": True, "age_ok": True},
        "benchmark_version": BENCHMARK_VERSION,
        "email_enabled":     EMAIL_ENABLED,
        "filename":          analysis.filename,
        "user":              user,
        "saved":             True,
    })


# ── Analyze ──────────────────────────────────────────────────────────────────

@app.post("/analyze", response_class=HTMLResponse)
async def analyze(
    request: Request,
    file: UploadFile = File(...),
    facility_name: str = Form(default=""),
    email: str = Form(default=""),
    db: Session = Depends(get_db),
):
    user = _get_current_user(request, db)

    if not file.filename.lower().endswith(".csv"):
        return _error(request, "Wrong file type",
                      "Please upload a CSV file (.csv). Export your dose data from "
                      "Radimetrics as CSV and try again.",
                      ["In Radimetrics: Reports → Dose Tracking → Export → CSV",
                       "Ensure the file extension is .csv, not .xlsx or .xls"], 400)

    contents = await file.read()
    if not contents:
        return _error(request, "Empty file",
                      "The uploaded file appears to be empty.",
                      ["Check that the export completed before downloading."], 400)

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp.write(contents); tmp_path = tmp.name

    try:
        with contextlib.redirect_stdout(io.StringIO()):
            df = load_radimetrics_csv(tmp_path)
    except Exception as e:
        Path(tmp_path).unlink(missing_ok=True)
        return _error(request, "Could not parse CSV", str(e),
                      ["Ensure the file is a valid comma-, tab-, or semicolon-separated CSV.",
                       "Try opening in Excel and re-saving as CSV (UTF-8).",
                       "Check that the file contains a DLP column and a study description or body region column."])
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    col_info = _column_report(df)

    if not col_info["dlp_ok"]:
        expected = ", ".join(COLUMN_MAPPINGS["dlp"])
        return _error(request, "DLP column not found",
                      f"A DLP column is required but was not found. Looked for: {expected}.",
                      [f"Columns found in your file: {', '.join(df.columns.tolist()[:12])}"])

    if not col_info["region_ok"]:
        return _error(request, "No body region information found",
                      "LeapfrogDose needs a Study Description or Body Region column.",
                      [f"Columns found: {', '.join(df.columns.tolist()[:12])}"])

    fname = facility_name.strip() or (user.facility_name if user else "") or "Facility"

    try:
        with contextlib.redirect_stdout(io.StringIO()):
            results = analyze_facility(df, fname)
    except Exception as e:
        return _error(request, "Analysis failed", str(e),
                      ["Ensure DLP values are numeric.",
                       "Check that age values are numbers (years), not strings like '45y'."], 500)

    # Save if logged in
    saved_id = None
    if user:
        record = AnalysisResult(
            user_id=user.id,
            facility_name=fname,
            filename=file.filename,
            results_json=json.dumps(results),
        )
        db.add(record); db.commit(); db.refresh(record)
        saved_id = record.id

    ctx = {
        "request":           request,
        "results":           results,
        "col_info":          col_info,
        "benchmark_version": BENCHMARK_VERSION,
        "email_enabled":     EMAIL_ENABLED,
        "filename":          file.filename,
        "user":              user,
        "saved":             saved_id is not None,
        "saved_id":          saved_id,
    }

    report_html = templates.TemplateResponse("report.html", ctx)

    if email.strip() and EMAIL_ENABLED:
        try:
            _send_report_email(email.strip(), fname,
                               report_html.body.decode() if hasattr(report_html, "body") else "")
        except Exception:
            pass

    return report_html
