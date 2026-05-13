# GammaMetric

**Physics-grounded AI reliability monitoring for CT lung nodule detection.**

Live product: [dose.gammametric.com](https://dose.gammametric.com)

---

## What It Does

GammaMetric tells you whether the acquisition conditions on a given CT scan are sufficient for your deployed AI to be trusted — before a radiologist signs the report.

Two tools, one pipeline:

**Tool 1 — Acquisition Alert System** (live)
Ingests CT studies via Orthanc webhook, reads DICOM headers, and classifies acquisition conditions GREEN / YELLOW / RED based on experimentally characterized sensitivity thresholds. A RED alert fires immediately; YELLOW flags accumulate into a daily digest. Protects against reading AI output on a compromised acquisition.

**Tool 2 — Longitudinal Comparability Layer** (live at `/longitudinal`)
Matches nodules across follow-up studies, computes diameter estimates, and attaches an acquisition reliability score to each comparison. Flags whether apparent interval change exceeds what protocol variation alone can explain. Validated on 262 nodules across 183 LIDC-IDRI cases.

Together: Tool 1 fires at ingestion. Tool 2 fires at read time. One tells you whether the acquisition is acceptable. The other tells you whether the longitudinal comparison is trustworthy.

---

## Key Validation Finding

Baseline → 5 mm slice thickness (GREEN → RED):
- Apparent nodule growth of **+164%** and **+179%** in a single case
- Two additional nodules missed entirely
- No biological change — acquisition protocol variation only

This is the problem the system is designed to catch.

---

## Live Demo

[dose.gammametric.com/longitudinal?case_id=LIDC-IDRI-0092&condition_a=baseline&condition_b=thick_5mm](https://dose.gammametric.com/longitudinal?case_id=LIDC-IDRI-0092&condition_a=baseline&condition_b=thick_5mm)

---

## Validation Basis

- Model: MONAI RetinaNet, LUNA16-trained
- Dataset: LIDC-IDRI (154 cases, 409 consensus nodules ≥3 mm, ≥3 readers)
- Conditions characterized: slice thickness (1.25–5 mm), dose (2.5–10 mGy), reconstruction kernel (STANDARD / SOFT)
- Anchor finding: 42% relative sensitivity drop for 3–6 mm nodules at 5 mm slice thickness
- Longitudinal validation: 262 nodules across 183 cases, mean bias −1.56 mm, 95% LoA [−5.41, +2.29 mm]

Preprint: [arXiv 2603.26785](https://arxiv.org/abs/2603.26785) — under review at *Academic Radiology*

---

## Repository Structure

```
core/
  sensitivity_engine.py    # Acquisition reliability scoring (GREEN/YELLOW/RED)
  alert_engine.py          # RED alert and YELLOW digest email delivery
  leapfrog_dose.py         # CT dose analytics engine (Leapfrog Section 8B)
  report_generator.py      # Dose report output

web/
  app.py                   # FastAPI app — dose.gammametric.com
  sensitivity_router.py    # /api/sensitivity — Orthanc webhook + compute endpoint
  longitudinal_router.py   # /longitudinal — nodule comparability tool
  templates/               # Jinja2 HTML templates

validation/
  nodule_longitudinal.py           # Longitudinal nodule matching engine
  degradation_engine.py            # Physics-guided image degradation simulation
  gammametric_validation_pipeline.py
  run_inference_batch.py
  batch_pipeline.py / v2
  consensus_batch.py
  annotation_pipeline.py
  threshold_sensitivity_analysis.py

data/
  nodule_results/          # Pre-computed MONAI inference results (183 cases × 6 conditions)
```

---

## API

**POST /api/sensitivity/compute** — score a single acquisition
```json
{
  "slice_thickness_mm": 5.0,
  "reconstruction_kernel": "STANDARD",
  "ctdivol_mgy": 10.0,
  "scanner_model": "GE Revolution",
  "model_version": "luna16_v1.0.0"
}
```

**POST /api/sensitivity/dicom/study** — Orthanc webhook target (DICOM tags → classification + alert)

**GET /api/longitudinal/compare?case_id=X&condition_a=Y&condition_b=Z** — JSON longitudinal comparison

---

## Deployment

Hosted on Railway. Deployed automatically on push to `master`.

```toml
# railway.toml
[deploy]
startCommand = "uvicorn web.app:app --host 0.0.0.0 --port $PORT"
```

Environment variables:
- `SECRET_KEY` — session signing key
- `DATABASE_URL` — PostgreSQL URL (defaults to SQLite)
- `RESEND_API_KEY` — email delivery for alerts
- `NODULE_RESULTS_DIR` — override path for inference results (defaults to `data/nodule_results/`)

---

## Contact

Dan Soliman, MS — [dan@gammametric.com](mailto:dan@gammametric.com)
