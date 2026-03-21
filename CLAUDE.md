# GammaMetric — Claude Code Context

## Project Overview
GammaMetric is an independent medical physics consultancy (Dan Soliman, ABR board-eligible)
focused on CT dose analytics, protocol optimization, and AI robustness validation for
outpatient imaging centers.

This repository contains two major components:
1. **LeapfrogDose** — CT dose analytics and Leapfrog Section 8B compliance reporting
2. **AI Validation Pipeline** — robustness validation of CT lung nodule detection AI
   under acquisition parameter variation (the "Beyond Benchmarks" preprint)

## Environment
- Python virtual environment: .venv/ (activate before running anything)
- Key packages: pandas, pydicom, monai, pylidc, matplotlib, numpy, scipy
- Full dependencies: requirements.txt

## Project Structure

core/
  leapfrog_dose.py         # Main LeapfrogDose analytics engine
  report_generator.py      # Leapfrog Section 8B report output
  generate_sample_data.py  # Synthetic data for testing

validation/
  gammametric_validation_pipeline.py  # Master pipeline
  degradation_engine.py               # Applies dose reduction / slice thickness changes
  batch_pipeline.py / v2              # Batch inference across cases
  run_inference_batch.py              # MONAI RetinaNet inference
  rerun_0009_inference.py             # Single-case rerun (LIDC-0009)
  annotation_pipeline.py             # Pulls LIDC-IDRI annotations via pylidc
  get_annotations.py                 # Annotation utilities
  consensus_batch.py                 # Consensus nodule detection logic
  fix_comparison.py                  # Post-hoc result comparison/repair
  check_volumes.py                   # Volume validation checks
  inventory_cases.py                 # Case inventory and status tracking
  plot_nodule_heatmap.py             # Heatmap figure generation
  slice_thickness_figure_v2.py       # Slice thickness sensitivity figure
  threshold_sensitivity_analysis.py  # Consensus threshold sensitivity (newest)
  make_carousel.py                   # Figure carousel for presentation
  regen_0009_dose.py                 # Regenerate dose variants for LIDC-0009
  patch_pylidc.py                    # pylidc compatibility patch
  download_lidc.py                   # LIDC-IDRI download utility

docs/
  Beyond Benchmarks *.docx / .pdf    # Preprint manuscript
  table1.docx                        # Results table
  fig1_sensitivity_bar.png           # Sensitivity by condition figure
  fig2_heatmap.png                   # Heatmap figure
  tanguay-et-al-2022-*.pdf          # Key reference

gui/                                 # GUI components (in development)
output/                              # Runtime outputs (not git-tracked)

## Data (stored outside repo)
NIfTI files and DICOM data live on the Desktop in gammametric_output/ and
gammametric_validation_*/ folders. These are not git-tracked due to size.

LIDC-IDRI cases used in preprint: LIDC-0009, LIDC-IDRI-0001, 0002, 0087, 0089,
0092, 0093, 0151.

## AI Validation — Key Facts
- Model: MONAI RetinaNet trained on LUNA16
- Dataset: LIDC-IDRI
- Experimental conditions: baseline, 25% dose reduction, 50% dose reduction,
  3mm slice thickness, 5mm slice thickness
- Key findings: ~4 percentage point sensitivity drop at 25% dose reduction;
  ~19 percentage point drop at 5mm slice thickness
- Consensus threshold sensitivity analysis: varies threshold from 1-4 annotators
- Framing: pilot methodology paper, not definitive clinical validation

## LeapfrogDose — Key Facts
- Targets Leapfrog Group Hospital Survey Section 8B CT dose reporting
- Uses routine-only exams for percentile calculations (non-routine filtered out)
- Benchmarks sourced from ACR Dose Index Registry (DIR) 2023-2024
- Key regions: Head, Chest, Abdomen-Pelvis, Pediatric
- Metric: DLP (dose length product, mGy cm)

## Domain Conventions
- Dose reduction variants named: _dose_25pct, _dose_50pct
- Slice thickness variants named: _thick_3mm, _thick_5mm
- NIfTI files follow LIDC-IDRI naming conventions
- Leapfrog reports use routine-only exam subsets per Section 8B spec

## Current Priorities
1. Preprint submitted to arXiv — monitor for feedback, prepare for journal submission
2. GammaMetric business development — RAI (Daytona Beach) top prospect

## Long-Term Vision
1. **SaaS AI detection pipeline** — productize the validation/inference pipeline
   into a deployable service for CT lung nodule detection, expanding to MRI and
   other modalities over time
2. **Budget-friendly dose analytics platform** — a community hospital alternative
   to Radimetrics/DoseWatch; LeapfrogDose is the foundation of this product
1. Finalize preprint — run sensitivity analyses at varying consensus thresholds
   before submission (threshold_sensitivity_analysis.py)
2. GammaMetric business development — RAI (Daytona Beach) top prospect
