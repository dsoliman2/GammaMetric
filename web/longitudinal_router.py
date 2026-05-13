import sys
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

sys.path.insert(0, str(Path(__file__).parent.parent))
from validation.nodule_longitudinal import (
    CONDITION_ACQ_PARAMS,
    RESULTS_BASE,
    compare_studies,
)

router = APIRouter(tags=["longitudinal"])
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")

CONDITIONS = list(CONDITION_ACQ_PARAMS.keys())
CONDITION_LABELS = {
    "baseline":    "Baseline (1.25 mm, 10 mGy)",
    "dose_25pct":  "25% Dose Reduction (7.5 mGy)",
    "dose_50pct":  "50% Dose Reduction (5.0 mGy)",
    "thick_3mm":   "3 mm Slice Thickness",
    "thick_5mm":   "5 mm Slice Thickness",
    "soft_kernel": "Soft Kernel (b40f)",
}


def _available_cases() -> list[str]:
    base = Path(RESULTS_BASE)
    if not base.exists():
        return []
    cases: set[str] = set()
    for d in base.iterdir():
        if not d.is_dir():
            continue
        for cond in CONDITIONS:
            if d.name.endswith(f"_{cond}"):
                case_id = d.name[: -(len(cond) + 1)]
                if (d / "result_luna16_fold0.json").exists():
                    cases.add(case_id)
                break
    return sorted(cases)


@router.get("/longitudinal", response_class=HTMLResponse)
def longitudinal_page(
    request: Request,
    case_id: Optional[str] = Query(None),
    condition_a: Optional[str] = Query(None),
    condition_b: Optional[str] = Query(None),
):
    cases = _available_cases()
    result = None
    error = None

    if case_id and condition_a and condition_b:
        if condition_a not in CONDITIONS or condition_b not in CONDITIONS:
            error = "Invalid condition specified."
        elif condition_a == condition_b:
            error = "Condition A and Condition B must be different."
        else:
            try:
                result = compare_studies(case_id, condition_a, condition_b)
            except Exception as exc:
                error = str(exc)

    return templates.TemplateResponse(request, "longitudinal.html", {
        "cases":            cases,
        "conditions":       CONDITIONS,
        "condition_labels": CONDITION_LABELS,
        "case_id":          case_id,
        "condition_a":      condition_a,
        "condition_b":      condition_b,
        "result":           result,
        "error":            error,
    })


@router.get("/api/longitudinal/cases")
def list_cases() -> dict:
    return {"cases": _available_cases(), "conditions": CONDITIONS, "condition_labels": CONDITION_LABELS}


@router.get("/api/longitudinal/compare")
def compare_api(
    case_id: str = Query(...),
    condition_a: str = Query(...),
    condition_b: str = Query(...),
) -> dict:
    if condition_a not in CONDITIONS or condition_b not in CONDITIONS:
        raise HTTPException(400, "Invalid condition name.")
    if condition_a == condition_b:
        raise HTTPException(400, "condition_a and condition_b must differ.")
    r = compare_studies(case_id, condition_a, condition_b)
    return {
        "case_id":             r.case_id,
        "condition_a":         r.condition_a,
        "condition_b":         r.condition_b,
        "comparable":          r.comparable,
        "comparability_note":  r.comparability_note,
        "acq_a":               r.acq_a,
        "acq_b":               r.acq_b,
        "matched": [
            {
                "nodule_id":          p.nodule_id,
                "diameter_a_mm":      p.study_a.diameter_mm,
                "diameter_b_mm":      p.study_b.diameter_mm,
                "diameter_delta_mm":  p.diameter_delta_mm,
                "diameter_delta_pct": p.diameter_delta_pct,
                "centroid_dist_mm":   p.centroid_dist_mm,
                "confidence_a":       p.study_a.confidence,
                "confidence_b":       p.study_b.confidence,
                "exceeds_noise":      p.exceeds_noise,
            }
            for p in r.matched
        ],
        "only_in_a": [
            {"centroid_mm": d.centroid_mm, "diameter_mm": d.diameter_mm, "confidence": d.confidence}
            for d in r.only_in_a
        ],
        "only_in_b": [
            {"centroid_mm": d.centroid_mm, "diameter_mm": d.diameter_mm, "confidence": d.confidence}
            for d in r.only_in_b
        ],
    }
