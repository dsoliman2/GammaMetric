"""
GammaMetric Alert Engine

RED  → immediate email on ingestion
YELLOW → queued; dispatched via daily digest endpoint
"""

from __future__ import annotations
import json
import os
import logging
from datetime import datetime, timedelta
from typing import Optional

import resend
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

RESEND_API_KEY = os.getenv("RESEND_API_KEY", os.getenv("SMTP_PASS", ""))
EMAIL_FROM     = os.getenv("EMAIL_FROM", "onboarding@resend.dev")
ALERT_TO       = os.getenv("ALERT_TO", "")
EMAIL_ENABLED  = bool(RESEND_API_KEY and ALERT_TO)

if RESEND_API_KEY:
    resend.api_key = RESEND_API_KEY


def _send(subject: str, body_text: str, body_html: str) -> bool:
    if not EMAIL_ENABLED:
        logger.warning("Email not configured — alert not sent. Set RESEND_API_KEY and ALERT_TO.")
        return False
    try:
        resend.Emails.send({
            "from":    f"GammaMetric <{EMAIL_FROM}>",
            "to":      [ALERT_TO],
            "subject": subject,
            "html":    body_html,
            "text":    body_text,
        })
        logger.info("Alert sent to %s: %s", ALERT_TO, subject)
        return True
    except Exception as e:
        logger.error("Failed to send alert: %s", e)
        return False


# ---------------------------------------------------------------------------
# RED — immediate
# ---------------------------------------------------------------------------

def _diameter_note(result_json_str: Optional[str]) -> str:
    """Returns an HTML note about diameter measurement uncertainty, or empty string."""
    if not result_json_str:
        return ""
    try:
        data = json.loads(result_json_str)
        du = data.get("diameter_uncertainty")
        if not du:
            return ""
        mean = du.get("mean_shift_mm", 0)
        ci   = du.get("uncertainty_95ci_mm", 0)
        if abs(mean) < 0.5 and ci < 2.0:
            return ""
        direction = "overestimated" if mean >= 0 else "underestimated"
        sign = "+" if mean >= 0 else ""
        return f"""
    <table width="100%" cellpadding="0" cellspacing="0" style="margin-bottom:16px">
     <tr><td style="background:#1a1a0f;border-left:3px solid #f59e0b;padding:12px 16px;border-radius:0 4px 4px 0">
      <p style="color:#fcd34d;font-size:12px;margin:0;line-height:1.6">
       <strong>Diameter measurement uncertainty:</strong>
       AI-reported nodule diameters may be {direction} by a mean of {sign}{mean:.1f}&nbsp;mm
       (95&nbsp;CI&nbsp;width&nbsp;{ci:.1f}&nbsp;mm) under current slice thickness.
       Longitudinal size comparisons should account for this acquisition-driven measurement variability.
      </p>
     </td></tr>
    </table>"""
    except Exception:
        return ""


def _driver_pills(result_json_str: Optional[str]) -> str:
    if not result_json_str:
        return ""
    try:
        data = json.loads(result_json_str)
        drivers = data.get("drivers", [])
        ci = data.get("confidence_interval", [])
        baseline = data.get("baseline_sensitivity", 0.782)
    except Exception:
        return ""
    if not drivers:
        return ""

    pill_style = (
        "display:inline-block;background:#3b0e0e;border:1px solid #7f1d1d;"
        "color:#fca5a5;font-size:11px;padding:4px 10px;border-radius:99px;margin:3px 3px 3px 0"
    )
    PARAM_LABELS = {
        "slice_thickness": "Slice thickness",
        "kernel": "Kernel",
        "dose": "CTDIvol",
    }
    pills = ""
    for d in drivers:
        raw_param = d.get("parameter", "")
        label = d.get("label") or PARAM_LABELS.get(raw_param, raw_param)
        delta = d.get("contribution_pp") or d.get("delta_pp", 0)
        if delta and abs(delta) >= 0.5:
            pills += f'<span style="{pill_style}">{label} &nbsp;{delta:+.1f}pp</span>'

    ci_str = ""
    if ci and len(ci) == 2:
        ci_str = f'<span style="color:#6b7280;font-size:11px"> &nbsp;CI [{ci[0]:.0%}–{ci[1]:.0%}]</span>'
    baseline_str = f'<div style="color:#6b7280;font-size:11px;margin-top:3px">baseline {baseline:.1%}{ci_str}</div>'

    return f"""
    <!-- Drivers -->
    <table width="100%" cellpadding="0" cellspacing="0" style="margin-bottom:20px">
     <tr><td>
      <div style="color:#6b7280;font-size:10px;text-transform:uppercase;letter-spacing:0.07em;margin-bottom:8px">Drivers</div>
      {pills}
     </td></tr>
    </table>
    """, baseline_str


def _red_html(study) -> str:
    ts = datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')
    sens_pct = f"{study.estimated_sensitivity:.0%}"
    result_json_str = study.result_json if hasattr(study, 'result_json') else None
    drivers_result = _driver_pills(result_json_str)
    drivers_html, baseline_html = drivers_result if isinstance(drivers_result, tuple) else ("", "")
    diam_note_html = _diameter_note(result_json_str)
    return f"""
<html><body style="margin:0;padding:0;background:#0a0e17;font-family:Arial,sans-serif">
<table width="100%" cellpadding="0" cellspacing="0">
 <tr><td align="center" style="padding:32px 16px">
  <table width="600" cellpadding="0" cellspacing="0" style="background:#111827;border-radius:8px;overflow:hidden">

   <!-- Header -->
   <tr><td style="background:#7f1d1d;padding:20px 28px">
    <table width="100%" cellpadding="0" cellspacing="0"><tr>
     <td>
      <span style="color:#fca5a5;font-size:11px;font-weight:bold;text-transform:uppercase;letter-spacing:0.1em">GammaMetric</span>
      <div style="color:#fff;font-size:19px;font-weight:bold;margin-top:5px">Acquisition Reliability Warning</div>
     </td>
     <td align="right" valign="middle">
      <span style="background:#ef4444;color:#fff;padding:5px 13px;border-radius:4px;font-size:12px;font-weight:bold;letter-spacing:0.06em">RED</span>
     </td>
    </tr></table>
   </td></tr>

   <!-- Body -->
   <tr><td style="padding:24px 28px">

    <!-- Scanner / date -->
    <table width="100%" cellpadding="0" cellspacing="0" style="margin-bottom:20px">
     <tr>
      <td style="color:#6b7280;font-size:11px;text-transform:uppercase;letter-spacing:0.07em;padding-bottom:3px">Scanner</td>
      <td align="right" style="color:#6b7280;font-size:11px;text-transform:uppercase;letter-spacing:0.07em;padding-bottom:3px">Date</td>
     </tr>
     <tr>
      <td style="color:#f9fafb;font-size:13px">{study.scanner_model or 'unknown'}</td>
      <td align="right" style="color:#f9fafb;font-size:13px">{study.acquisition_date or '—'}</td>
     </tr>
    </table>

    <!-- Metric tiles -->
    <table width="100%" cellpadding="0" cellspacing="0" style="margin-bottom:20px">
     <tr>
      <td width="32%" style="background:#1f2937;padding:14px 12px;border-radius:6px">
       <div style="color:#6b7280;font-size:10px;text-transform:uppercase;letter-spacing:0.07em;margin-bottom:5px">Est. sensitivity</div>
       <div style="color:#f9fafb;font-size:22px;font-weight:bold">{sens_pct}</div>
       {baseline_html}
      </td>
      <td width="4%"></td>
      <td width="32%" style="background:#1f2937;padding:14px 12px;border-radius:6px">
       <div style="color:#6b7280;font-size:10px;text-transform:uppercase;letter-spacing:0.07em;margin-bottom:5px">Reduction</div>
       <div style="color:#ef4444;font-size:22px;font-weight:bold">−{study.degradation_pp:.1f}pp</div>
      </td>
      <td width="4%"></td>
      <td width="28%" style="background:#1f2937;padding:14px 12px;border-radius:6px">
       <div style="color:#6b7280;font-size:10px;text-transform:uppercase;letter-spacing:0.07em;margin-bottom:5px">Slice</div>
       <div style="color:#f9fafb;font-size:22px;font-weight:bold">{study.slice_thickness_mm}mm</div>
      </td>
     </tr>
    </table>

    <!-- Acquisition params -->
    <table width="100%" cellpadding="0" cellspacing="0" style="margin-bottom:20px">
     <tr>
      <td style="color:#9ca3af;font-size:12px;padding:7px 0;border-bottom:1px solid #1f2937">Kernel</td>
      <td align="right" style="color:#f9fafb;font-size:12px;padding:7px 0;border-bottom:1px solid #1f2937">{study.reconstruction_kernel}</td>
     </tr>
     <tr>
      <td style="color:#9ca3af;font-size:12px;padding:7px 0;border-bottom:1px solid #1f2937">CTDIvol</td>
      <td align="right" style="color:#f9fafb;font-size:12px;padding:7px 0;border-bottom:1px solid #1f2937">{study.ctdivol_mgy} mGy</td>
     </tr>
     <tr>
      <td style="color:#9ca3af;font-size:12px;padding:7px 0">Study UID</td>
      <td align="right" style="color:#6b7280;font-size:11px;padding:7px 0;word-break:break-all">{study.study_instance_uid}</td>
     </tr>
    </table>

    {drivers_html}

    <!-- Clinical note -->
    <table width="100%" cellpadding="0" cellspacing="0" style="margin-bottom:16px">
     <tr><td style="background:#1a0f0f;border-left:3px solid #ef4444;padding:14px 16px;border-radius:0 4px 4px 0">
      <p style="color:#fca5a5;font-size:13px;margin:0;line-height:1.65">
       Estimated AI sensitivity for 3–6 mm nodules is reduced under current acquisition conditions.
       Interpret longitudinal AI comparisons with caution.
      </p>
     </td></tr>
    </table>

    {diam_note_html}

    <!-- Footer -->
    <p style="color:#374151;font-size:11px;margin:0">GammaMetric AI Reliability Monitor &mdash; {ts}</p>

   </td></tr>
  </table>
 </td></tr>
</table>
</body></html>"""


def send_red_alert(study, db: Session) -> bool:
    subject = f"[GammaMetric] Acquisition Reliability Warning — {study.degradation_pp:.1f}pp sensitivity reduction"
    diam_text = ""
    try:
        if study.result_json:
            du = json.loads(study.result_json).get("diameter_uncertainty", {})
            if du and (abs(du.get("mean_shift_mm", 0)) >= 0.5 or du.get("uncertainty_95ci_mm", 0) >= 2.0):
                diam_text = (
                    f"\nDiameter uncertainty: AI diameters may be shifted by "
                    f"{du['mean_shift_mm']:+.1f}mm (95 CI width {du['uncertainty_95ci_mm']:.1f}mm) "
                    f"under current slice thickness.\n"
                )
    except Exception:
        pass
    text = (
        f"Acquisition Reliability Warning\n\n"
        f"Study UID: {study.study_instance_uid}\n"
        f"Sensitivity reduction: -{study.degradation_pp:.1f}pp\n"
        f"Estimated sensitivity: {study.estimated_sensitivity:.0%}\n"
        f"Slice: {study.slice_thickness_mm}mm | Kernel: {study.reconstruction_kernel} | CTDIvol: {study.ctdivol_mgy}mGy\n\n"
        f"Estimated AI sensitivity for 3-6mm nodules is reduced under current acquisition conditions. "
        f"Interpret longitudinal AI comparisons with caution.{diam_text}\n\n"
        f"GammaMetric AI Reliability Monitor"
    )
    sent = _send(subject, text, _red_html(study))
    if sent:
        study.alerted = True
        db.commit()
    return sent


# ---------------------------------------------------------------------------
# YELLOW — daily digest
# ---------------------------------------------------------------------------

def _digest_html(studies: list) -> str:
    ts = datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')
    count = len(studies)
    rows = ""
    for s in studies:
        rows += f"""
     <tr>
      <td style="padding:9px 10px;border-bottom:1px solid #1f2937;color:#9ca3af;font-size:12px">{s.acquisition_date or '—'}</td>
      <td style="padding:9px 10px;border-bottom:1px solid #1f2937;color:#6b7280;font-size:11px;word-break:break-all">{s.study_instance_uid}</td>
      <td style="padding:9px 10px;border-bottom:1px solid #1f2937;color:#f9fafb;font-size:12px">{s.slice_thickness_mm}mm</td>
      <td style="padding:9px 10px;border-bottom:1px solid #1f2937;color:#f9fafb;font-size:12px">{s.reconstruction_kernel}</td>
      <td style="padding:9px 10px;border-bottom:1px solid #1f2937;color:#f9fafb;font-size:12px">{s.ctdivol_mgy}mGy</td>
      <td style="padding:9px 10px;border-bottom:1px solid #1f2937;color:#fbbf24;font-size:12px;font-weight:bold">−{s.degradation_pp:.1f}pp</td>
     </tr>"""
    return f"""
<html><body style="margin:0;padding:0;background:#0a0e17;font-family:Arial,sans-serif">
<table width="100%" cellpadding="0" cellspacing="0">
 <tr><td align="center" style="padding:32px 16px">
  <table width="620" cellpadding="0" cellspacing="0" style="background:#111827;border-radius:8px;overflow:hidden">

   <!-- Header -->
   <tr><td style="background:#78350f;padding:20px 28px">
    <table width="100%" cellpadding="0" cellspacing="0"><tr>
     <td>
      <span style="color:#fcd34d;font-size:11px;font-weight:bold;text-transform:uppercase;letter-spacing:0.1em">GammaMetric</span>
      <div style="color:#fff;font-size:19px;font-weight:bold;margin-top:5px">Daily Reliability Digest</div>
     </td>
     <td align="right" valign="middle">
      <span style="background:#f59e0b;color:#111;padding:5px 13px;border-radius:4px;font-size:12px;font-weight:bold;letter-spacing:0.06em">YELLOW &times;{count}</span>
     </td>
    </tr></table>
   </td></tr>

   <!-- Body -->
   <tr><td style="padding:24px 28px">
    <p style="color:#9ca3af;font-size:13px;margin:0 0 20px">
     {count} {'study' if count == 1 else 'studies'} with moderately reduced AI sensitivity (5–10pp) in the past 24 hours.
     No immediate action required; consider reviewing acquisition protocol trends.
    </p>

    <!-- Table -->
    <table width="100%" cellpadding="0" cellspacing="0" style="border-radius:6px;overflow:hidden">
     <tr style="background:#1f2937">
      <td style="padding:9px 10px;color:#6b7280;font-size:10px;text-transform:uppercase;letter-spacing:0.07em">Date</td>
      <td style="padding:9px 10px;color:#6b7280;font-size:10px;text-transform:uppercase;letter-spacing:0.07em">Study UID</td>
      <td style="padding:9px 10px;color:#6b7280;font-size:10px;text-transform:uppercase;letter-spacing:0.07em">Slice</td>
      <td style="padding:9px 10px;color:#6b7280;font-size:10px;text-transform:uppercase;letter-spacing:0.07em">Kernel</td>
      <td style="padding:9px 10px;color:#6b7280;font-size:10px;text-transform:uppercase;letter-spacing:0.07em">CTDIvol</td>
      <td style="padding:9px 10px;color:#6b7280;font-size:10px;text-transform:uppercase;letter-spacing:0.07em">Reduction</td>
     </tr>
     {rows}
    </table>

    <p style="color:#374151;font-size:11px;margin:20px 0 0">GammaMetric AI Reliability Monitor &mdash; {ts}</p>
   </td></tr>
  </table>
 </td></tr>
</table>
</body></html>"""


def send_yellow_digest(db: Session, lookback_hours: int = 24) -> dict:
    """
    Query all unalerted YELLOW studies from the past lookback_hours, send digest, mark alerted.
    Returns summary dict.
    """
    from web.models import StudyResult
    cutoff = datetime.utcnow() - timedelta(hours=lookback_hours)
    studies = (
        db.query(StudyResult)
        .filter(
            StudyResult.classification == "YELLOW",
            StudyResult.alerted == False,
            StudyResult.ingested_at >= cutoff,
        )
        .order_by(StudyResult.ingested_at.desc())
        .all()
    )

    if not studies:
        return {"sent": False, "reason": "no unalerted YELLOW studies in window", "count": 0}

    subject = f"[GammaMetric] Daily Digest — {len(studies)} YELLOW {'study' if len(studies)==1 else 'studies'}"
    text = (
        f"YELLOW digest — {len(studies)} studies with moderate sensitivity degradation (5-10pp) "
        f"in the past {lookback_hours} hours.\n\n"
        + "\n".join(
            f"- {s.study_instance_uid}: -{s.degradation_pp:.1f}pp "
            f"({s.slice_thickness_mm}mm, {s.reconstruction_kernel}, {s.ctdivol_mgy}mGy)"
            for s in studies
        )
        + "\n\nGammaMetric AI Reliability Monitor"
    )
    sent = _send(subject, text, _digest_html(studies))
    if sent:
        for s in studies:
            s.alerted = True
        db.commit()

    return {"sent": sent, "count": len(studies)}
