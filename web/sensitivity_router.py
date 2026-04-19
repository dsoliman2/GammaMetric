import json
import sys
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends
from fastapi.responses import Response
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).parent.parent))
from core.sensitivity_engine import compute, SensitivityInput
from core.alert_engine import send_red_alert, send_yellow_digest
from core.pdf_report import generate_report
from web.models import get_db, StudyResult

router = APIRouter(prefix="/api/sensitivity", tags=["sensitivity"])


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class SensitivityRequest(BaseModel):
    slice_thickness_mm: float    = Field(..., gt=0)
    reconstruction_kernel: str   = Field(...)
    ctdivol_mgy: float           = Field(..., gt=0)
    scanner_model: str           = Field("unknown")
    model_version: str           = Field("unknown")


class DicomWebhookPayload(BaseModel):
    """DICOM tags extracted by Orthanc and forwarded to this endpoint."""
    study_instance_uid:    str            = Field(..., alias="StudyInstanceUID")
    slice_thickness_mm:    float          = Field(..., alias="SliceThickness")
    reconstruction_kernel: str            = Field(..., alias="ConvolutionKernel")
    ctdivol_mgy:           float          = Field(..., alias="CTDIvol")
    scanner_model:         str            = Field("unknown", alias="ManufacturerModelName")
    kvp:                   Optional[float] = Field(None, alias="KVP")
    acquisition_date:      Optional[str]  = Field(None, alias="AcquisitionDate")
    model_version:         str            = Field("unknown", alias="AIModelVersion")

    class Config:
        populate_by_name = True


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("/compute")
def compute_sensitivity(req: SensitivityRequest) -> dict:
    result = compute(SensitivityInput(
        slice_thickness_mm=req.slice_thickness_mm,
        reconstruction_kernel=req.reconstruction_kernel,
        ctdivol_mgy=req.ctdivol_mgy,
        scanner_model=req.scanner_model,
        model_version=req.model_version,
    ))
    return result.to_dict()


@router.post("/dicom/study")
def ingest_dicom_study(payload: DicomWebhookPayload, db: Session = Depends(get_db)) -> dict:
    """
    Orthanc webhook target. Receives extracted DICOM tags, runs sensitivity engine,
    persists the result, returns classification.
    """
    result = compute(SensitivityInput(
        slice_thickness_mm=payload.slice_thickness_mm,
        reconstruction_kernel=payload.reconstruction_kernel,
        ctdivol_mgy=payload.ctdivol_mgy,
        scanner_model=payload.scanner_model,
        model_version=payload.model_version,
    ))

    row = StudyResult(
        study_instance_uid=payload.study_instance_uid,
        acquisition_date=payload.acquisition_date,
        scanner_model=payload.scanner_model,
        model_version=payload.model_version,
        slice_thickness_mm=payload.slice_thickness_mm,
        reconstruction_kernel=payload.reconstruction_kernel,
        ctdivol_mgy=payload.ctdivol_mgy,
        kvp=payload.kvp,
        estimated_sensitivity=result.estimated_sensitivity,
        degradation_pp=result.degradation_pp,
        classification=result.classification,
        out_of_distribution=result.out_of_distribution,
        result_json=json.dumps(result.to_dict()),
    )

    # Upsert: if we've seen this UID before (re-send), update in place.
    existing = db.query(StudyResult).filter_by(
        study_instance_uid=payload.study_instance_uid
    ).first()
    if existing:
        db.delete(existing)
        db.flush()

    db.add(row)
    db.commit()
    db.refresh(row)

    if result.classification == "RED":
        send_red_alert(row, db)

    return {
        "study_instance_uid":  payload.study_instance_uid,
        "classification":      result.classification,
        "degradation_pp":      result.degradation_pp,
        "out_of_distribution": result.out_of_distribution,
        "alert_sent":          row.alerted,
    }


@router.post("/report")
def generate_pdf_report(req: SensitivityRequest) -> Response:
    inp = SensitivityInput(
        slice_thickness_mm=req.slice_thickness_mm,
        reconstruction_kernel=req.reconstruction_kernel,
        ctdivol_mgy=req.ctdivol_mgy,
        scanner_model=req.scanner_model,
        model_version=req.model_version,
    )
    pdf_bytes = generate_report(inp)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=GammaMetric_Reliability_Report.pdf"},
    )


@router.post("/alerts/digest")
def trigger_digest(db: Session = Depends(get_db)) -> dict:
    """Trigger the YELLOW daily digest manually (call this from a cron job)."""
    return send_yellow_digest(db)
