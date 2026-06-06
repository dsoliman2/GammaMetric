import json
import sys
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends
from fastapi.responses import Response
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).parent.parent))
from core.sensitivity_engine import compute, SensitivityInput, DetectionInput
from core.alert_engine import send_red_alert, send_yellow_digest
from core.pdf_report import generate_report
from core.asir_detector import check_dicom_tags, ASIRResult
from web.models import get_db, StudyResult

router = APIRouter(prefix="/api/sensitivity", tags=["sensitivity"])


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class DetectionPayload(BaseModel):
    confidence:    float           = Field(..., ge=0.0, le=1.0)
    diameter_mm:   Optional[float] = None
    detection_id:  Optional[str]   = None


class SensitivityRequest(BaseModel):
    slice_thickness_mm: float    = Field(..., gt=0)
    reconstruction_kernel: str   = Field(...)
    ctdivol_mgy: float           = Field(..., gt=0)
    scanner_model: str           = Field("unknown")
    model_version: str           = Field("unknown")
    prior_kernel:  Optional[str] = Field(None)
    detections:    Optional[list[DetectionPayload]] = Field(None)


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
    prior_kernel:          Optional[str]            = Field(None, alias="PriorKernel")
    detections:            Optional[list[DetectionPayload]] = Field(None, alias="Detections")
    # Iterative recon: Orthanc can forward GE private tag (0053,1043) if extracted
    iterative_recon_tag:   Optional[str]  = Field(None, alias="IterativeReconLevel")

    class Config:
        populate_by_name = True


def _to_engine_detections(items: Optional[list[DetectionPayload]]) -> Optional[list[DetectionInput]]:
    if not items:
        return None
    return [DetectionInput(confidence=d.confidence,
                           diameter_mm=d.diameter_mm,
                           detection_id=d.detection_id) for d in items]


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
        prior_kernel=req.prior_kernel,
        detections=_to_engine_detections(req.detections),
    ))
    return result.to_dict()


@router.post("/dicom/study")
def ingest_dicom_study(payload: DicomWebhookPayload, db: Session = Depends(get_db)) -> dict:
    """
    Orthanc webhook target. Receives extracted DICOM tags (and optionally AI
    detections + prior kernel for detection-aware reliability), runs sensitivity
    engine, persists the result, returns classification.
    """
    result = compute(SensitivityInput(
        slice_thickness_mm=payload.slice_thickness_mm,
        reconstruction_kernel=payload.reconstruction_kernel,
        ctdivol_mgy=payload.ctdivol_mgy,
        scanner_model=payload.scanner_model,
        model_version=payload.model_version,
        prior_kernel=payload.prior_kernel,
        detections=_to_engine_detections(payload.detections),
    ))

    # ASIR detection — tag-based (pixel fallback requires DICOM dir access)
    asir: Optional[ASIRResult] = None
    if payload.iterative_recon_tag:
        # Orthanc forwarded the GE private tag directly
        asir = ASIRResult(
            detected=bool(payload.iterative_recon_tag.strip()),
            level=payload.iterative_recon_tag.strip() or None,
            confidence=1.0,
            method='private_tag',
            private_tag_value=payload.iterative_recon_tag,
        )
    else:
        # Fall back to kernel-suffix heuristic (Siemens SAFIRE)
        from core.asir_detector import _check_kernel_suffix
        asir = _check_kernel_suffix(payload.reconstruction_kernel)

    # Inject ASIR info into result_json so alert engine can render the block
    result_dict = result.to_dict()
    if asir and asir.detected:
        result_dict['iterative_recon'] = {
            'detected':          asir.detected,
            'level':             asir.level,
            'confidence':        asir.confidence,
            'method':            asir.method,
            'private_tag_value': asir.private_tag_value,
        }

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
        result_json=json.dumps(result_dict),
        iterative_recon_detected=asir.detected if asir else False,
        iterative_recon_level=asir.level if asir else None,
        iterative_recon_method=asir.method if asir else None,
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
