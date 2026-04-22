# GammaMetric — Physics-Guided AI Validation Pipeline

Code repository for: **"Effect of CT Acquisition Parameters on RetinaNet-Based Lung Nodule Detection: A Physics-Guided Validation Study Across Dose, Slice Thickness, and Reconstruction Kernel"** (Academic Radiology, under review)

## Overview

This repository contains the analysis pipeline used to evaluate the sensitivity of a deployed deep learning lung nodule detection model (MONAI RetinaNet, pretrained on LUNA16) to systematic variation in CT acquisition parameters. Six imaging conditions were evaluated: baseline, 25% dose reduction, 50% dose reduction, 3 mm slice thickness, 5 mm slice thickness, and soft reconstruction kernel. Image degradation was applied using physics-guided, image-domain simulation methods requiring no raw projection data or proprietary scanner access.

## Repository Structure

```
validation/
  degradation_engine.py              # Dose, slice thickness, and kernel simulation
  gammametric_validation_pipeline.py # Master pipeline (end-to-end)
  run_inference_batch.py             # MONAI RetinaNet batch inference
  batch_pipeline.py / v2             # Batch processing across cases
  consensus_batch.py                 # LIDC-IDRI consensus nodule extraction
  annotation_pipeline.py             # DICOM XML annotation parsing
  threshold_sensitivity_analysis.py  # Sensitivity analysis across confidence thresholds
  plot_nodule_heatmap.py             # Figure generation (heatmap)
  slice_thickness_figure_v2.py       # Figure generation (slice thickness sensitivity)

core/
  leapfrog_dose.py                   # LeapfrogDose CT dose analytics engine
  report_generator.py                # Leapfrog Section 8B report output

web/
  app.py                             # FastAPI web app (dose.gammametric.com)
```

## Data

The LIDC-IDRI dataset is publicly available through [The Cancer Imaging Archive](https://www.cancerimagingarchive.net). This pipeline was validated on 154 cases (409 consensus nodules, ≥3 readers, ≥3 mm).

## Requirements

```
pip install -r requirements.txt      # Core dependencies
pip install -r requirements-ml.txt  # ML/inference dependencies (MONAI, PyTorch)
```

## Usage

1. Download LIDC-IDRI cases via `validation/download_lidc.py`
2. Extract consensus nodule annotations via `validation/annotation_pipeline.py`
3. Run degradation and inference via `validation/gammametric_validation_pipeline.py`
4. Analyze threshold sensitivity via `validation/threshold_sensitivity_analysis.py`

## Citation

If you use this pipeline, please cite the associated manuscript (citation to be updated upon publication).

## Contact

Dan Soliman, MS — dan@gammametric.com
