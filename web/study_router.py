"""
Reader study router — /study
Pilot: acquisition-aware AI reliability flagging vs. standard presentation.
"""

import json
import random
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Form, Request, Depends
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from web.models import get_db, ReaderStudyParticipant, ReaderStudyResponse

router = APIRouter(tags=["study"])
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")

# ── Case definitions ──────────────────────────────────────────────────────────
# Each case: prior = baseline (1.25 mm), follow-up = thick_5mm (5 mm).
# comparability: GREEN (both detected, large), YELLOW (borderline), RED (missed at 5 mm).

STUDY_CASES = [
    # ── Group 1: 1.25 mm follow-up — acquisition identical to prior ───────────
    {
        "case_id": "LIDC-IDRI-0006", "nodule_idx": 1, "diam_mm": 8.4,
        "fu_recon_mm": 1.25, "fu_recon_label": "1.25 mm",
        "prior_det": True,  "fu_det": True,  "prior_conf": 0.91, "fu_conf": 0.90,
        "comparability": "GREEN",
        "detail": "Follow-up uses identical 1.25 mm reconstruction. Acquisition is directly comparable to prior. AI sensitivity within expected range.",
    },
    {
        "case_id": "LIDC-IDRI-0011", "nodule_idx": 0, "diam_mm": 9.4,
        "fu_recon_mm": 1.25, "fu_recon_label": "1.25 mm",
        "prior_det": True,  "fu_det": True,  "prior_conf": 0.94, "fu_conf": 0.93,
        "comparability": "GREEN",
        "detail": "Follow-up uses identical 1.25 mm reconstruction. Acquisition is directly comparable to prior.",
    },
    {
        "case_id": "LIDC-IDRI-0012", "nodule_idx": 2, "diam_mm": 7.1,
        "fu_recon_mm": 1.25, "fu_recon_label": "1.25 mm",
        "prior_det": True,  "fu_det": True,  "prior_conf": 0.87, "fu_conf": 0.86,
        "comparability": "GREEN",
        "detail": "Follow-up uses identical 1.25 mm reconstruction. Acquisition is directly comparable to prior.",
    },
    {
        "case_id": "LIDC-IDRI-0005", "nodule_idx": 1, "diam_mm": 7.3,
        "fu_recon_mm": 1.25, "fu_recon_label": "1.25 mm",
        "prior_det": True,  "fu_det": True,  "prior_conf": 0.89, "fu_conf": 0.88,
        "comparability": "GREEN",
        "detail": "Follow-up uses identical 1.25 mm reconstruction. Acquisition is directly comparable to prior.",
    },
    # ── Group 2: 2.5 mm follow-up — comparable for medium/large nodules ───────
    {
        "case_id": "LIDC-IDRI-0008", "nodule_idx": 1, "diam_mm": 7.8,
        "fu_recon_mm": 2.5, "fu_recon_label": "2.5 mm",
        "prior_det": True,  "fu_det": True,  "prior_conf": 0.88, "fu_conf": 0.84,
        "comparability": "GREEN",
        "detail": "Follow-up used 2.5 mm reconstruction. d/D = 0.32 for a 7.8 mm nodule — nodule adequately sampled. AI sensitivity within expected range for this acquisition.",
    },
    {
        "case_id": "LIDC-IDRI-0039", "nodule_idx": 3, "diam_mm": 6.6,
        "fu_recon_mm": 2.5, "fu_recon_label": "2.5 mm",
        "prior_det": True,  "fu_det": True,  "prior_conf": 0.83, "fu_conf": 0.79,
        "comparability": "GREEN",
        "detail": "Follow-up used 2.5 mm reconstruction. d/D = 0.38 for a 6.6 mm nodule — adequately sampled at this reconstruction thickness.",
    },
    {
        "case_id": "LIDC-IDRI-0046", "nodule_idx": 5, "diam_mm": 6.7,
        "fu_recon_mm": 2.5, "fu_recon_label": "2.5 mm",
        "prior_det": True,  "fu_det": True,  "prior_conf": 0.82, "fu_conf": 0.78,
        "comparability": "GREEN",
        "detail": "Follow-up used 2.5 mm reconstruction. d/D = 0.37 for a 6.7 mm nodule — adequately sampled. Acquisition supports reliable longitudinal comparison.",
    },
    {
        "case_id": "LIDC-IDRI-0004", "nodule_idx": 0, "diam_mm": 6.6,
        "fu_recon_mm": 2.5, "fu_recon_label": "2.5 mm",
        "prior_det": True,  "fu_det": True,  "prior_conf": 0.84, "fu_conf": 0.80,
        "comparability": "GREEN",
        "detail": "Follow-up used 2.5 mm reconstruction. d/D = 0.38 for a 6.6 mm nodule — adequately sampled at this reconstruction thickness.",
    },
    # ── Group 3: 5 mm follow-up — borderline or unreliable AI comparison ──────
    {
        "case_id": "LIDC-IDRI-0035", "nodule_idx": 0, "diam_mm": 4.9,
        "fu_recon_mm": 5.0, "fu_recon_label": "5 mm",
        "prior_det": True,  "fu_det": True,  "prior_conf": 0.78, "fu_conf": 0.57,
        "comparability": "YELLOW",
        "detail": "Follow-up used 5 mm reconstruction. d/D = 1.02 for a 4.9 mm nodule — at the undersampled threshold. AI sensitivity estimated at 74% (vs 91% at standard). A positive detection does not exclude interval growth; a miss at this d/D would not confirm regression.",
    },
    {
        "case_id": "LIDC-IDRI-0012", "nodule_idx": 5, "diam_mm": 5.2,
        "fu_recon_mm": 5.0, "fu_recon_label": "5 mm",
        "prior_det": True,  "fu_det": True,  "prior_conf": 0.81, "fu_conf": 0.61,
        "comparability": "YELLOW",
        "detail": "Follow-up used 5 mm reconstruction. d/D = 0.96 for a 5.2 mm nodule — approaching the undersampled boundary. AI sensitivity estimated at 76% (vs 91% at standard). Longitudinal comparison has reduced reliability at this acquisition.",
    },
    {
        "case_id": "LIDC-IDRI-0039", "nodule_idx": 0, "diam_mm": 4.2,
        "fu_recon_mm": 5.0, "fu_recon_label": "5 mm",
        "prior_det": True,  "fu_det": True,  "prior_conf": 0.74, "fu_conf": 0.53,
        "comparability": "YELLOW",
        "detail": "Follow-up used 5 mm reconstruction. d/D = 1.19 for a 4.2 mm nodule — undersampled regime. AI sensitivity estimated at 63% (vs 91% at standard). Detection at follow-up is reassuring but the low confidence and acquisition constraints limit the strength of this comparison.",
    },
    {
        "case_id": "LIDC-IDRI-0008", "nodule_idx": 0, "diam_mm": 6.6,
        "fu_recon_mm": 5.0, "fu_recon_label": "5 mm",
        "prior_det": True,  "fu_det": False, "prior_conf": 0.85, "fu_conf": None,
        "comparability": "RED",
        "detail": "Follow-up used 5 mm reconstruction. d/D = 0.76 for a 6.6 mm nodule. AI sensitivity estimated at 68% (vs 91% at standard). Non-detection is consistent with acquisition-related sensitivity loss; true regression cannot be confirmed from this series alone.",
    },
    {
        "case_id": "LIDC-IDRI-0034", "nodule_idx": 0, "diam_mm": 4.0,
        "fu_recon_mm": 5.0, "fu_recon_label": "5 mm",
        "prior_det": True,  "fu_det": False, "prior_conf": 0.71, "fu_conf": None,
        "comparability": "RED",
        "detail": "Follow-up used 5 mm reconstruction. d/D = 1.25 for a 4.0 mm nodule — undersampled regime. Estimated AI sensitivity 59% (vs 91%). Non-detection is highly consistent with an acquisition-related miss for a nodule of this size.",
    },
    {
        "case_id": "LIDC-IDRI-0039", "nodule_idx": 1, "diam_mm": 3.4,
        "fu_recon_mm": 5.0, "fu_recon_label": "5 mm",
        "prior_det": True,  "fu_det": False, "prior_conf": 0.65, "fu_conf": None,
        "comparability": "RED",
        "detail": "Follow-up used 5 mm reconstruction. d/D = 1.47 for a 3.4 mm nodule — well into the undersampled regime. Estimated AI sensitivity 55% (vs 91%). Non-detection at this acquisition is the expected outcome for a nodule of this size; true regression cannot be inferred.",
    },
    {
        "case_id": "LIDC-IDRI-0012", "nodule_idx": 3, "diam_mm": 7.4,
        "fu_recon_mm": 5.0, "fu_recon_label": "5 mm",
        "prior_det": True,  "fu_det": False, "prior_conf": 0.86, "fu_conf": None,
        "comparability": "RED",
        "detail": "Follow-up used 5 mm reconstruction. d/D = 0.68 for a 7.4 mm nodule. Estimated AI sensitivity 70% (vs 91%). Non-detection at follow-up is not sufficient to confirm interval regression given this acquisition.",
    },
]

N_CASES = len(STUDY_CASES)


def _make_token() -> str:
    return uuid.uuid4().hex


def _make_assignment() -> tuple[list[int], dict[str, bool]]:
    """Random case order + random flag assignment (~half flagged)."""
    order = list(range(N_CASES))
    random.shuffle(order)
    flagged = set(random.sample(order, N_CASES // 2))
    assignment = {str(i): (i in flagged) for i in range(N_CASES)}
    return order, assignment


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/study", response_class=HTMLResponse)
def study_landing(request: Request):
    return templates.TemplateResponse(request, "study_landing.html", {
        "n_cases": N_CASES,
    })


@router.post("/study/enroll", response_class=HTMLResponse)
def study_enroll(
    request: Request,
    name: str = Form(...),
    role: str = Form(...),
    institution: str = Form(default=""),
    db: Session = Depends(get_db),
):
    name = name.strip()
    role = role.strip()
    if not name or not role:
        return templates.TemplateResponse(request, "study_landing.html", {
            "n_cases": N_CASES,
            "error": "Please enter your name and role before beginning.",
        })

    token = _make_token()
    order, assignment = _make_assignment()

    participant = ReaderStudyParticipant(
        token=token,
        name=name,
        role=role,
        institution=institution.strip(),
        case_order=json.dumps(order),
        flag_assignment=json.dumps(assignment),
    )
    db.add(participant)
    db.commit()

    return RedirectResponse(f"/study/{token}/case/1", status_code=302)


@router.get("/study/{token}/case/{n}", response_class=HTMLResponse)
def study_case(
    token: str,
    n: int,
    request: Request,
    db: Session = Depends(get_db),
):
    participant = db.get(ReaderStudyParticipant, token)
    if not participant:
        return RedirectResponse("/study", status_code=302)
    if participant.completed_at:
        return RedirectResponse(f"/study/{token}/complete", status_code=302)
    if not 1 <= n <= N_CASES:
        return RedirectResponse(f"/study/{token}/complete", status_code=302)

    order = json.loads(participant.case_order)
    assignment = json.loads(participant.flag_assignment)

    case_idx = order[n - 1]
    case = STUDY_CASES[case_idx]
    show_flag = assignment[str(case_idx)]

    img_base = f"/static/study_images/{case['case_id']}_n{case['nodule_idx']}"
    fu_recon = case.get("fu_recon_mm", 5.0)
    if fu_recon <= 1.25:
        fu_img_suffix = "_prior"
    elif fu_recon <= 2.5:
        fu_img_suffix = "_fu25"
    else:
        fu_img_suffix = "_fu"
    return templates.TemplateResponse(request, "study_case.html", {
        "token":      token,
        "n":          n,
        "total":      N_CASES,
        "case":       case,
        "show_flag":  show_flag,
        "case_idx":   case_idx,
        "progress":   round(100 * (n - 1) / N_CASES),
        "img_prior":  img_base + "_prior.png",
        "img_fu":     img_base + fu_img_suffix + ".png",
    })


@router.post("/study/{token}/case/{n}", response_class=HTMLResponse)
def study_case_submit(
    token: str,
    n: int,
    request: Request,
    case_idx: int = Form(...),
    show_flag: bool = Form(...),
    management_decision: str = Form(...),
    confidence_rating: int = Form(default=0),
    response_time_sec: float = Form(default=0.0),
    db: Session = Depends(get_db),
):
    participant = db.get(ReaderStudyParticipant, token)
    if not participant:
        return RedirectResponse("/study", status_code=302)

    valid_decisions = {"surveillance", "shorten", "escalate", "thin_slice"}
    if management_decision not in valid_decisions:
        return RedirectResponse(f"/study/{token}/case/{n}", status_code=302)

    # Upsert response (allow re-submission if participant goes back)
    existing = (
        db.query(ReaderStudyResponse)
        .filter_by(participant_token=token, case_idx=case_idx)
        .first()
    )
    conf = confidence_rating if 1 <= confidence_rating <= 5 else None
    if existing:
        existing.management_decision = management_decision
        existing.confidence_rating = conf
        existing.response_time_sec = response_time_sec
        existing.show_flag = show_flag
    else:
        db.add(ReaderStudyResponse(
            participant_token=token,
            case_idx=case_idx,
            show_flag=show_flag,
            management_decision=management_decision,
            confidence_rating=conf,
            response_time_sec=response_time_sec,
        ))
    db.commit()

    if n < N_CASES:
        return RedirectResponse(f"/study/{token}/case/{n + 1}", status_code=302)

    # Final case — mark complete
    participant.completed_at = datetime.utcnow()
    db.commit()
    return RedirectResponse(f"/study/{token}/complete", status_code=302)


@router.get("/study/{token}/complete", response_class=HTMLResponse)
def study_complete(token: str, request: Request, db: Session = Depends(get_db)):
    participant = db.get(ReaderStudyParticipant, token)
    n_completed = (
        db.query(ReaderStudyResponse)
        .filter_by(participant_token=token)
        .count()
    ) if participant else 0
    return templates.TemplateResponse(request, "study_complete.html", {
        "participant": participant,
        "n_completed": n_completed,
        "n_total":     N_CASES,
    })


@router.get("/study/admin/results")
def study_results(request: Request, db: Session = Depends(get_db)):
    """Raw TSV export — no auth for now (URL security only)."""
    rows = db.query(ReaderStudyResponse).all()
    lines = ["participant_token\tcase_idx\tcase_id\tdiam_mm\tfu_recon_mm\tcomparability\tfu_det\tshow_flag\tdecision\tconfidence\ttime_sec"]
    for r in rows:
        c = STUDY_CASES[r.case_idx]
        lines.append("\t".join([
            r.participant_token[:8],
            str(r.case_idx),
            c["case_id"],
            str(c["diam_mm"]),
            str(c.get("fu_recon_mm", 5.0)),
            c["comparability"],
            str(c["fu_det"]),
            str(r.show_flag),
            r.management_decision,
            str(r.confidence_rating) if r.confidence_rating else "",
            f"{r.response_time_sec:.1f}" if r.response_time_sec else "0",
        ]))
    from fastapi.responses import PlainTextResponse
    return PlainTextResponse("\n".join(lines),
                             headers={"Content-Disposition": "attachment; filename=study_results.tsv"})
