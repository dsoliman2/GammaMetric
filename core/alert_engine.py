"""
GammaMetric Alert Engine

RED  → immediate email on ingestion
YELLOW → queued; dispatched via daily digest endpoint
"""

from __future__ import annotations
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

def _red_html(study) -> str:
    return f"""
<html><body style="font-family:Arial,sans-serif;max-width:600px;margin:auto">
<div style="background:#c0392b;color:white;padding:16px;border-radius:4px">
  <h2 style="margin:0">&#9888; RED Alert — AI Sensitivity Critically Degraded</h2>
</div>
<div style="padding:16px;border:1px solid #ddd;margin-top:8px;border-radius:4px">
  <p><strong>Study UID:</strong> {study.study_instance_uid}</p>
  <p><strong>Date:</strong> {study.acquisition_date or 'unknown'}</p>
  <p><strong>Scanner:</strong> {study.scanner_model or 'unknown'}</p>
  <table style="border-collapse:collapse;width:100%">
    <tr style="background:#f5f5f5">
      <td style="padding:8px;border:1px solid #ddd"><strong>Estimated Sensitivity</strong></td>
      <td style="padding:8px;border:1px solid #ddd">{study.estimated_sensitivity:.1%}</td>
    </tr>
    <tr>
      <td style="padding:8px;border:1px solid #ddd"><strong>Degradation</strong></td>
      <td style="padding:8px;border:1px solid #ddd;color:#c0392b"><strong>−{study.degradation_pp:.1f}pp</strong></td>
    </tr>
    <tr style="background:#f5f5f5">
      <td style="padding:8px;border:1px solid #ddd"><strong>Slice Thickness</strong></td>
      <td style="padding:8px;border:1px solid #ddd">{study.slice_thickness_mm} mm</td>
    </tr>
    <tr>
      <td style="padding:8px;border:1px solid #ddd"><strong>Kernel</strong></td>
      <td style="padding:8px;border:1px solid #ddd">{study.reconstruction_kernel}</td>
    </tr>
    <tr style="background:#f5f5f5">
      <td style="padding:8px;border:1px solid #ddd"><strong>CTDIvol</strong></td>
      <td style="padding:8px;border:1px solid #ddd">{study.ctdivol_mgy} mGy</td>
    </tr>
  </table>
  <p style="margin-top:16px;padding:12px;background:#fdf2f2;border-left:4px solid #c0392b">
    Sensitivity for 3–6mm nodules is critically degraded under current acquisition parameters.
    Review protocol immediately. Primary concern: missed nodules in the clinically critical detection window.
  </p>
  <p style="color:#666;font-size:12px">GammaMetric AI Reliability Monitor &mdash; {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}</p>
</div>
</body></html>"""


def send_red_alert(study, db: Session) -> bool:
    subject = f"[RED] AI Sensitivity Alert — {study.degradation_pp:.1f}pp degradation detected"
    text = (
        f"RED ALERT — AI sensitivity critically degraded.\n\n"
        f"Study UID: {study.study_instance_uid}\n"
        f"Degradation: -{study.degradation_pp:.1f}pp\n"
        f"Estimated sensitivity: {study.estimated_sensitivity:.1%}\n"
        f"Slice: {study.slice_thickness_mm}mm | Kernel: {study.reconstruction_kernel} | CTDIvol: {study.ctdivol_mgy}mGy\n\n"
        f"Review acquisition protocol immediately.\n\n"
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
    rows = ""
    for s in studies:
        rows += f"""
        <tr>
          <td style="padding:8px;border:1px solid #ddd">{s.acquisition_date or '—'}</td>
          <td style="padding:8px;border:1px solid #ddd">{s.study_instance_uid}</td>
          <td style="padding:8px;border:1px solid #ddd">{s.slice_thickness_mm}mm</td>
          <td style="padding:8px;border:1px solid #ddd">{s.reconstruction_kernel}</td>
          <td style="padding:8px;border:1px solid #ddd">{s.ctdivol_mgy}mGy</td>
          <td style="padding:8px;border:1px solid #ddd;color:#e67e22"><strong>−{s.degradation_pp:.1f}pp</strong></td>
        </tr>"""
    return f"""
<html><body style="font-family:Arial,sans-serif;max-width:700px;margin:auto">
<div style="background:#e67e22;color:white;padding:16px;border-radius:4px">
  <h2 style="margin:0">&#9888; Daily Digest — {len(studies)} YELLOW {"Study" if len(studies)==1 else "Studies"}</h2>
</div>
<div style="padding:16px;border:1px solid #ddd;margin-top:8px;border-radius:4px">
  <p>The following studies showed moderate AI sensitivity degradation (5–10pp) in the past 24 hours.
  No immediate action required, but review protocol trends.</p>
  <table style="border-collapse:collapse;width:100%;margin-top:12px">
    <tr style="background:#f5f5f5;font-weight:bold">
      <td style="padding:8px;border:1px solid #ddd">Date</td>
      <td style="padding:8px;border:1px solid #ddd">Study UID</td>
      <td style="padding:8px;border:1px solid #ddd">Slice</td>
      <td style="padding:8px;border:1px solid #ddd">Kernel</td>
      <td style="padding:8px;border:1px solid #ddd">CTDIvol</td>
      <td style="padding:8px;border:1px solid #ddd">Degradation</td>
    </tr>
    {rows}
  </table>
  <p style="color:#666;font-size:12px;margin-top:16px">
    GammaMetric AI Reliability Monitor &mdash; {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}
  </p>
</div>
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
