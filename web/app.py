"""
GammaDose Web App
====================
FastAPI app — upload a Radimetrics CSV, get a Leapfrog-ready HTML report.
Accounts + history storage for Phase 1 platform.

Email delivery: set SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS env vars to enable.
"""

import os
import sys
import json
import logging
import asyncio
import tempfile
import contextlib
import io

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("leapfrogdose")
from pathlib import Path


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

import numpy as np
import resend
import pandas as pd
from fastapi import FastAPI, File, UploadFile, Form, Request, Depends
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).parent.parent))
from core.leapfrog_dose import load_radimetrics_csv, analyze_facility, COLUMN_MAPPINGS, BENCHMARK_VERSION
from web.models import init_db, get_db, User, AnalysisResult, DriftAlert
from web.auth import (hash_password, verify_password, set_session, clear_session,
                      get_user_id, make_reset_token, verify_reset_token)
from web.sensitivity_router import router as sensitivity_router
from web.longitudinal_router import router as longitudinal_router
from web.study_router import router as study_router

# ── Email config ────────────────────────────────────────────────────────────
RESEND_API_KEY = os.getenv("RESEND_API_KEY", os.getenv("SMTP_PASS", ""))
EMAIL_FROM     = os.getenv("EMAIL_FROM", "onboarding@resend.dev")
EMAIL_ENABLED  = bool(RESEND_API_KEY)
if RESEND_API_KEY:
    resend.api_key = RESEND_API_KEY

# ── App setup ───────────────────────────────────────────────────────────────
app = FastAPI(title="GammaDose", docs_url=None, redoc_url=None)
app.include_router(sensitivity_router)
app.include_router(longitudinal_router)
app.include_router(study_router)
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
        request, "error.html",
        {"title": title, "message": message, "suggestions": suggestions or []},
        status_code=status,
    )


def _send_email(to_address: str, subject: str, html: str, text: str):
    resend.Emails.send({
        "from": f"GammaDose <{EMAIL_FROM}>",
        "to": [to_address],
        "subject": subject,
        "html": html,
        "text": text,
    })


def _send_report_email(to_address: str, facility_name: str, html: str):
    _send_email(
        to_address,
        subject=f"GammaDose Report — {facility_name}",
        html=html,
        text=f"Your dose report for {facility_name} is ready.\n\n— GammaDose by GammaMetric",
    )


def _check_drift(old_results: dict, new_results: dict) -> list[dict]:
    """Compare two analysis results dicts. Return list of drifted regions."""
    drifts = []
    old_adult = old_results.get("adult", {})
    new_adult = new_results.get("adult", {})
    for region, new_data in new_adult.items():
        new_status = new_data["benchmark_comparison"]["status"]
        new_p75    = new_data["stats"].get("p75", 0)
        old_data   = old_adult.get(region)
        if old_data is None:
            continue
        old_status = old_data["benchmark_comparison"]["status"]
        old_p75    = old_data["stats"].get("p75", 0)

        newly_above = (old_status != "ABOVE BENCHMARK" and new_status == "ABOVE BENCHMARK")
        worsened    = (old_status == "ABOVE BENCHMARK" and new_status == "ABOVE BENCHMARK"
                       and old_p75 > 0 and (new_p75 - old_p75) / old_p75 >= 0.15)
        if newly_above or worsened:
            drifts.append({
                "region":     region,
                "old_status": old_status,
                "new_status": new_status,
                "old_p75":    old_p75,
                "new_p75":    new_p75,
                "newly_above": newly_above,
            })
    return drifts


def _send_drift_alert(to_address: str, facility_name: str, drifts: list[dict]):
    rows_html = ""
    rows_text = ""
    for d in drifts:
        change = f"{d['old_p75']:.0f} → {d['new_p75']:.0f} mGy·cm"
        flag   = "NEW ▲" if d["newly_above"] else "WORSE ▲"
        rows_html += (
            f"<tr><td style='padding:6px 12px;border-bottom:1px solid #e2e8f0'><b>{d['region']}</b></td>"
            f"<td style='padding:6px 12px;border-bottom:1px solid #e2e8f0'>{change}</td>"
            f"<td style='padding:6px 12px;border-bottom:1px solid #e2e8f0;color:#dc2626'>{flag}</td></tr>"
        )
        rows_text += f"  {d['region']}: {change} [{flag}]\n"

    html = f"""
<div style="font-family:sans-serif;max-width:560px;margin:0 auto">
  <h2 style="color:#0f2342">Dose drift detected — {facility_name}</h2>
  <p style="color:#475569">One or more CT dose regions have crossed above the ACR DIR 75th percentile
  benchmark since your last upload.</p>
  <table style="width:100%;border-collapse:collapse;margin:1.5rem 0;font-size:14px">
    <thead>
      <tr style="background:#f1f5f9">
        <th style="padding:8px 12px;text-align:left">Region</th>
        <th style="padding:8px 12px;text-align:left">P75 Change</th>
        <th style="padding:8px 12px;text-align:left">Flag</th>
      </tr>
    </thead>
    <tbody>{rows_html}</tbody>
  </table>
  <p style="color:#475569">Log in to GammaDose to review your full report and protocol recommendations.</p>
  <p style="color:#94a3b8;font-size:12px">— GammaDose by GammaMetric</p>
</div>"""

    text = (
        f"Dose drift detected — {facility_name}\n\n"
        f"The following regions have crossed above the ACR DIR benchmark:\n\n"
        f"{rows_text}\n"
        "Log in to GammaDose to review your full report.\n\n"
        "— GammaDose by GammaMetric"
    )
    _send_email(to_address, f"⚠ Dose drift detected — {facility_name}", html, text)


# ── Public routes ────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request, db: Session = Depends(get_db)):
    user = _get_current_user(request, db)
    return templates.TemplateResponse(
        request, "index.html",
        {"email_enabled": EMAIL_ENABLED, "user": user},
    )


@app.get("/template")
async def download_template():
    template_path = Path(__file__).parent / "dose_template.csv"
    return StreamingResponse(
        open(template_path, "rb"),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=dose_template.csv"},
    )


@app.get("/signup", response_class=HTMLResponse)
async def signup_get(request: Request, db: Session = Depends(get_db)):
    if _get_current_user(request, db):
        return RedirectResponse("/dashboard", status_code=302)
    return templates.TemplateResponse(request, "signup.html", {"error": None})


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
            request, "signup.html",
            {"error": "An account with that email already exists."},
        )
    if len(password) < 8:
        return templates.TemplateResponse(
            request, "signup.html",
            {"error": "Password must be at least 8 characters."},
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
    return templates.TemplateResponse(request, "login.html", {"error": None})


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
            request, "login.html",
            {"error": "Incorrect email or password."},
        )
    response = RedirectResponse("/dashboard", status_code=302)
    set_session(response, user.id)
    return response


@app.get("/logout")
async def logout(request: Request):
    response = RedirectResponse("/", status_code=302)
    clear_session(response)
    return response


@app.get("/forgot-password", response_class=HTMLResponse)
async def forgot_password_get(request: Request):
    return templates.TemplateResponse(request, "forgot_password.html", {"sent": False, "error": None})


@app.post("/forgot-password", response_class=HTMLResponse)
async def forgot_password_post(
    request: Request,
    email: str = Form(...),
    db: Session = Depends(get_db),
):
    email = email.strip().lower()
    user = db.query(User).filter(User.email == email).first()
    logger.info("forgot-password: email=%s user_found=%s email_enabled=%s", email, bool(user), EMAIL_ENABLED)
    if user and EMAIL_ENABLED:
        token = make_reset_token(email)
        reset_url = str(request.base_url).rstrip("/") + f"/reset-password?token={token}"
        try:
            reset_html = (
                f'<p>Click the link below to reset your password. The link expires in 1 hour.</p>'
                f'<p><a href="{reset_url}">{reset_url}</a></p>'
                f'<p>If you did not request this, ignore this email.</p>'
                f'<p>— GammaDose by GammaMetric</p>'
            )
            reset_text = (
                f"Reset your password here (expires in 1 hour):\n\n{reset_url}\n\n"
                "If you did not request this, ignore this email.\n\n— GammaDose by GammaMetric"
            )
            await asyncio.to_thread(_send_email, email, "GammaDose — Password Reset", reset_html, reset_text)
            logger.info("forgot-password: email sent to %s", email)
        except Exception as exc:
            logger.error("forgot-password: Resend error: %s", exc)
    # Always show the same message to avoid email enumeration
    return templates.TemplateResponse(request, "forgot_password.html", {"sent": True, "error": None})


@app.get("/reset-password", response_class=HTMLResponse)
async def reset_password_get(request: Request, token: str = ""):
    email = verify_reset_token(token)
    if not email:
        return templates.TemplateResponse(
            request, "reset_password.html",
            {"token": token, "invalid": True, "error": None, "success": False},
        )
    return templates.TemplateResponse(
        request, "reset_password.html",
        {"token": token, "invalid": False, "error": None, "success": False},
    )


@app.post("/reset-password", response_class=HTMLResponse)
async def reset_password_post(
    request: Request,
    token: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db),
):
    email = verify_reset_token(token)
    if not email:
        return templates.TemplateResponse(
            request, "reset_password.html",
            {"token": token, "invalid": True, "error": None, "success": False},
        )
    if len(password) < 8:
        return templates.TemplateResponse(
            request, "reset_password.html",
            {"token": token, "invalid": False, "error": "Password must be at least 8 characters.", "success": False},
        )
    user = db.query(User).filter(User.email == email).first()
    if not user:
        return templates.TemplateResponse(
            request, "reset_password.html",
            {"token": token, "invalid": True, "error": None, "success": False},
        )
    user.password_hash = hash_password(password)
    db.commit()
    return templates.TemplateResponse(
        request, "reset_password.html",
        {"token": token, "invalid": False, "error": None, "success": True},
    )


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

    pending_alerts = (
        db.query(DriftAlert)
        .filter(DriftAlert.user_id == user.id, DriftAlert.acknowledged_at == None)
        .order_by(DriftAlert.created_at.desc())
        .all()
    )

    return templates.TemplateResponse(request, "dashboard.html", {
        "user":           user,
        "analyses":       analyses,
        "trend":          trend,
        "pending_alerts": pending_alerts,
    })


@app.post("/alerts/{alert_id}/acknowledge", response_class=HTMLResponse)
async def acknowledge_alert(
    alert_id: int,
    request: Request,
    note: str = Form(default=""),
    db: Session = Depends(get_db),
):
    user = _get_current_user(request, db)
    if not user:
        return RedirectResponse("/login", status_code=302)
    alert = db.query(DriftAlert).filter(
        DriftAlert.id == alert_id,
        DriftAlert.user_id == user.id,
    ).first()
    if alert:
        from datetime import datetime as dt
        alert.acknowledged_at = dt.utcnow()
        alert.acknowledged_by = user.email
        alert.note = note.strip() or None
        db.commit()
    return RedirectResponse("/dashboard", status_code=302)


@app.get("/alerts/log", response_class=HTMLResponse)
async def alerts_log(request: Request, db: Session = Depends(get_db)):
    user = _get_current_user(request, db)
    if not user:
        return RedirectResponse("/login", status_code=302)
    alerts = (
        db.query(DriftAlert)
        .filter(DriftAlert.user_id == user.id)
        .order_by(DriftAlert.created_at.desc())
        .all()
    )
    return templates.TemplateResponse(request, "alerts_log.html", {
        "user": user, "alerts": alerts,
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
    return templates.TemplateResponse(request, "report.html", {
        "results":           analysis.results,
        "col_info":          {"found": [], "critical_missing": [], "optional_missing": [],
                              "region_ok": True, "dlp_ok": True, "age_ok": True},
        "benchmark_version": BENCHMARK_VERSION,
        "email_enabled":     EMAIL_ENABLED,
        "filename":          analysis.filename,
        "user":              user,
        "saved":             True,
    })


# ── Excel Export ─────────────────────────────────────────────────────────────

def _results_to_excel(results: dict) -> io.BytesIO:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        # Adult sheet
        adult_rows = []
        for region, data in results.get("adult", {}).items():
            s = data["stats"]
            b = data["benchmark_comparison"]
            adult_rows.append({
                "Region": region,
                "N": s.get("count"),
                "P25 (mGy·cm)": s.get("p25"),
                "Median P50 (mGy·cm)": s.get("p50"),
                "P75 (mGy·cm)": s.get("p75"),
                "DIR Benchmark P75": b.get("benchmark_p75"),
                "Status": b.get("status"),
            })
        if adult_rows:
            pd.DataFrame(adult_rows).to_excel(writer, sheet_name="Adult CT", index=False)

        # Pediatric sheet
        peds_rows = []
        for region, age_groups in results.get("pediatric", {}).items():
            for age_grp, data in age_groups.items():
                s = data.get("stats", {})
                peds_rows.append({
                    "Region": region,
                    "Age Group": age_grp,
                    "N": s.get("count"),
                    "Median P50 (mGy·cm)": s.get("p50"),
                    "Min N Met": data.get("min_n_met", ""),
                })
        if peds_rows:
            pd.DataFrame(peds_rows).to_excel(writer, sheet_name="Pediatric CT", index=False)

        # Outliers sheet
        outliers = results.get("outliers", [])
        if outliers:
            pd.DataFrame(outliers).to_excel(writer, sheet_name="Outliers", index=False)

    buf.seek(0)
    return buf


@app.get("/export/{analysis_id}")
async def export_excel(analysis_id: int, request: Request, db: Session = Depends(get_db)):
    user = _get_current_user(request, db)
    if not user:
        return RedirectResponse("/login", status_code=302)
    analysis = db.query(AnalysisResult).filter(
        AnalysisResult.id == analysis_id,
        AnalysisResult.user_id == user.id,
    ).first()
    if not analysis:
        return _error(request, "Report not found", "This report doesn't exist or doesn't belong to your account.", status=404)

    buf = _results_to_excel(analysis.results)
    fname = (analysis.facility_name or "report").replace(" ", "_")
    return StreamingResponse(
        buf,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename={fname}_dose_report.xlsx"},
    )


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
                      "GammaDose needs a Study Description or Body Region column.",
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
        # Grab previous analysis before saving the new one
        prev_analysis = user.analyses[0] if user.analyses else None

        record = AnalysisResult(
            user_id=user.id,
            facility_name=fname,
            filename=file.filename,
            results_json=json.dumps(results, cls=_NumpyEncoder),
        )
        db.add(record); db.commit(); db.refresh(record)
        saved_id = record.id

        # Drift alert — compare to previous upload
        if prev_analysis:
            drifts = _check_drift(prev_analysis.results, results)
            if drifts:
                for d in drifts:
                    db.add(DriftAlert(
                        user_id=user.id,
                        analysis_id=record.id,
                        region=d["region"],
                        old_status=d["old_status"],
                        new_status=d["new_status"],
                        old_p75=d["old_p75"],
                        new_p75=d["new_p75"],
                    ))
                db.commit()
                if EMAIL_ENABLED:
                    asyncio.create_task(
                        asyncio.to_thread(_send_drift_alert, user.email, fname, drifts)
                    )
                logger.info("drift-alert: %d region(s) drifted for %s", len(drifts), user.email)

    ctx = {
        "results":           results,
        "col_info":          col_info,
        "benchmark_version": BENCHMARK_VERSION,
        "email_enabled":     EMAIL_ENABLED,
        "filename":          file.filename,
        "user":              user,
        "saved":             saved_id is not None,
        "saved_id":          saved_id,
    }

    report_html = templates.TemplateResponse(request, "report.html", ctx)

    if email.strip() and EMAIL_ENABLED:
        try:
            _send_report_email(email.strip(), fname,
                               report_html.body.decode() if hasattr(report_html, "body") else "")
        except Exception:
            pass

    return report_html
