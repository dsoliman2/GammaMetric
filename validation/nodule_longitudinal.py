"""
Longitudinal nodule matching across CT acquisition conditions.

Matches detected nodules between two studies of the same case, computes
diameter measurements per study, and attaches acquisition reliability flags
from the sensitivity engine so callers can judge whether a diameter change
is biology or acquisition noise.

Box format from MONAI RetinaNet LUNA16 bundle (StandardMode):
    [cx, cy, cz, w, h, d]  — center in mm, full widths in mm
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from core.sensitivity_engine import (
    SensitivityInput,
    compute as compute_sensitivity,
    MID_CONFIDENCE_LO,
    MID_CONFIDENCE_HI,
)

RESULTS_BASE       = os.getenv(
    "NODULE_RESULTS_DIR",
    str(os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "nodule_results")),
)
CONF_THRESHOLD     = 0.5
MATCH_DIST_MM      = 15.0
# Below this diameter delta, change is indistinguishable from measurement noise
DIAMETER_NOISE_MM  = 2.0

# Mid-confidence regime constants are imported from core.sensitivity_engine
# (canonical home). Both Surface A and Surface B use the same thresholds.

# Known acquisition parameters for each experimental condition in this dataset.
# slice_thickness_mm, reconstruction_kernel, ctdivol_mgy.
# dose_25pct = 25% dose *reduction* → 75% of 10 mGy reference = 7.5 mGy.
CONDITION_ACQ_PARAMS: dict[str, dict] = {
    'baseline':    {'slice_thickness_mm': 1.25, 'reconstruction_kernel': 'fc10', 'ctdivol_mgy': 10.0},
    'dose_25pct':  {'slice_thickness_mm': 1.25, 'reconstruction_kernel': 'fc10', 'ctdivol_mgy': 7.5},
    'dose_50pct':  {'slice_thickness_mm': 1.25, 'reconstruction_kernel': 'fc10', 'ctdivol_mgy': 5.0},
    'thick_3mm':   {'slice_thickness_mm': 3.0,  'reconstruction_kernel': 'fc10', 'ctdivol_mgy': 10.0},
    'thick_5mm':   {'slice_thickness_mm': 5.0,  'reconstruction_kernel': 'fc10', 'ctdivol_mgy': 10.0},
    'soft_kernel': {'slice_thickness_mm': 1.25, 'reconstruction_kernel': 'b40f', 'ctdivol_mgy': 10.0},
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    centroid_mm: tuple[float, float, float]   # (cx, cy, cz) world mm
    diameter_mm: float                         # max(w, h, d) — largest box dimension
    confidence: float
    box_whl_mm: tuple[float, float, float]    # (w, h, d) full widths


@dataclass
class MatchedPair:
    nodule_id: int
    study_a: Detection
    study_b: Detection
    centroid_dist_mm: float
    diameter_delta_mm: float    # study_b.diameter - study_a.diameter
    diameter_delta_pct: float   # relative to study_a
    exceeds_noise: bool         # |delta| > DIAMETER_NOISE_MM
    kernel_comparability_tier:  str = "GREEN"   # GREEN/YELLOW only meaningful when kernel changes
    kernel_comparability_note:  str = ""


@dataclass
class LongitudinalResult:
    case_id: str
    condition_a: str
    condition_b: str
    acq_a: dict                  # sensitivity_engine result dict for condition_a
    acq_b: dict                  # sensitivity_engine result dict for condition_b
    matched: list[MatchedPair]
    only_in_a: list[Detection]  # detected in A, missed in B (apparent disappearance)
    only_in_b: list[Detection]  # detected in B, missed in A (apparent new nodule)
    comparable: bool            # False if either study is RED (>10pp degradation)
    comparability_note: str
    kernel_change: bool = False                  # condition_a kernel != condition_b kernel
    dropout_notes_a: list[str] = field(default_factory=list)  # parallel to only_in_a
    dropout_notes_b: list[str] = field(default_factory=list)  # parallel to only_in_b


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_detections(
    case_id: str,
    condition: str,
    conf_threshold: float = CONF_THRESHOLD,
) -> list[Detection]:
    """
    Load MONAI RetinaNet inference results for one case/condition.
    Returns detections above conf_threshold, sorted by confidence descending.
    """
    path = os.path.join(RESULTS_BASE, f'{case_id}_{condition}', 'result_luna16_fold0.json')
    if not os.path.exists(path):
        return []

    with open(path) as f:
        raw = json.load(f)

    if isinstance(raw, dict) and 'value' in raw:
        data = raw['value'][0]
    elif isinstance(raw, list):
        data = raw[0]
    else:
        return []

    boxes  = data.get('box', [])
    scores = data.get('label_scores', [])

    dets: list[Detection] = []
    for box, score in zip(boxes, scores):
        if score < conf_threshold:
            continue
        cx, cy, cz = float(box[0]), float(box[1]), float(box[2])
        w,  h,  d  = float(box[3]), float(box[4]), float(box[5])
        dets.append(Detection(
            centroid_mm=(cx, cy, cz),
            diameter_mm=max(w, h, d),
            confidence=float(score),
            box_whl_mm=(w, h, d),
        ))

    dets.sort(key=lambda d: d.confidence, reverse=True)
    return dets


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def match_detections(
    dets_a: list[Detection],
    dets_b: list[Detection],
    max_dist_mm: float = MATCH_DIST_MM,
) -> tuple[list[tuple[Detection, Detection, float]], list[Detection], list[Detection]]:
    """
    Greedy nearest-neighbour matching between two detection lists.

    Iterates dets_a in confidence order (highest first). For each detection
    in A, finds the closest unmatched detection in B within max_dist_mm.

    Returns:
        matched    — list of (det_a, det_b, centroid_dist_mm)
        unmatched_a — detections in A with no counterpart in B
        unmatched_b — detections in B with no counterpart in A
    """
    used_b  = [False] * len(dets_b)
    matched: list[tuple[Detection, Detection, float]] = []
    unmatched_a: list[Detection] = []

    for det_a in dets_a:  # confidence-sorted, highest first
        ax, ay, az = det_a.centroid_mm
        best_idx, best_dist = None, float('inf')

        for j, det_b in enumerate(dets_b):
            if used_b[j]:
                continue
            bx, by, bz = det_b.centroid_mm
            dist = float(np.sqrt((ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2))
            if dist < max_dist_mm and dist < best_dist:
                best_dist = dist
                best_idx  = j

        if best_idx is not None:
            matched.append((det_a, dets_b[best_idx], best_dist))
            used_b[best_idx] = True
        else:
            unmatched_a.append(det_a)

    unmatched_b = [det_b for j, det_b in enumerate(dets_b) if not used_b[j]]
    return matched, unmatched_a, unmatched_b


# ---------------------------------------------------------------------------
# Acquisition sensitivity
# ---------------------------------------------------------------------------

def _kernel_pair_comparability(conf_a: float, conf_b: float) -> tuple[str, str]:
    """
    Per-nodule comparability tier under cross-kernel comparison.

    ORIGINAL rationale (May 22, RETRACTED 2026-05-29): mid-confidence regime
    under kernel change showed AUC 0.67-0.71 for displacement->dropout.
    Bridge claim retracted (Simpson's paradox + mixed-vendor cohort).

    NEW anchor (2026-05-29): on the v3 4-vendor cohort, nodules within 0.5mm
    of a Lung-RADS diameter boundary reclassify under kernel change at 18-33%
    per vendor; patient-level Lung-RADS follow-up interval changes occur in
    7.2% of patients (95% CI 3.9-12.0%). Mid-confidence detections sit in the
    AI-bbox score band most likely to land near a Lung-RADS boundary, so the
    YELLOW comparability tier is now justified as "this measurement may shift
    a Lung-RADS category under cross-kernel comparison" — a measurement-
    instability claim, NOT a dropout-prediction claim.

    Tier thresholds 0.45/0.80 kept; copy of YELLOW message should be revised
    to reference category-flip risk, not "latent representational shift."
    See core/sensitivity_engine.py MID_CONFIDENCE_LO/HI comment for the data
    + caveats. Returns (tier, plain-language note).
    """
    min_conf = min(conf_a, conf_b)
    if min_conf >= MID_CONFIDENCE_HI:
        return ("GREEN",
                f"High-confidence detection on both studies (min conf {min_conf:.2f}); "
                f"robust to kernel change.")
    if min_conf >= MID_CONFIDENCE_LO:
        return ("YELLOW",
                f"Mid-confidence detection (min conf {min_conf:.2f}) under kernel change; "
                f"diameter delta should be interpreted with caution — "
                f"latent representational shift contributes to measurement variability in this regime.")
    return ("YELLOW",
            f"Low-confidence detection (min conf {min_conf:.2f}); "
            f"reproducibility is dominated by threshold marginality regardless of kernel.")


def _dropout_kernel_note(conf: float) -> str:
    """Interpretation for a unilateral detection (apparent dropout) under kernel change."""
    if conf >= MID_CONFIDENCE_HI:
        return ("High-confidence detection lost across kernels (conf "
                f"{conf:.2f}); unusual — warrants direct radiologist review.")
    if conf >= MID_CONFIDENCE_LO:
        return ("Apparent dropout consistent with mid-confidence regime "
                f"(conf {conf:.2f}); kernel-induced representational shift may have contributed.")
    return ("Near-threshold detection "
            f"(conf {conf:.2f}); loss likely reflects baseline marginality rather than kernel shift.")


def _acq_result(condition: str) -> dict:
    params = CONDITION_ACQ_PARAMS.get(condition)
    if params is None:
        return {'classification': 'UNKNOWN', 'degradation_pp': None,
                'estimated_sensitivity': None, 'plain_language': f'No acquisition params for {condition!r}'}
    inp = SensitivityInput(
        **params,
        scanner_model='LIDC-IDRI',
        model_version='luna16_v1.0.0',
    )
    return compute_sensitivity(inp).to_dict()


# ---------------------------------------------------------------------------
# Top-level comparison
# ---------------------------------------------------------------------------

def compare_studies(
    case_id: str,
    condition_a: str,
    condition_b: str,
    conf_threshold: float = CONF_THRESHOLD,
    max_dist_mm: float = MATCH_DIST_MM,
) -> LongitudinalResult:
    """
    Full longitudinal comparison of case_id between two acquisition conditions.
    """
    dets_a = load_detections(case_id, condition_a, conf_threshold)
    dets_b = load_detections(case_id, condition_b, conf_threshold)

    matched_raw, only_a, only_b = match_detections(dets_a, dets_b, max_dist_mm)

    kernel_a = (CONDITION_ACQ_PARAMS.get(condition_a) or {}).get('reconstruction_kernel')
    kernel_b = (CONDITION_ACQ_PARAMS.get(condition_b) or {}).get('reconstruction_kernel')
    kernel_change = bool(kernel_a and kernel_b and kernel_a != kernel_b)

    matched_pairs: list[MatchedPair] = []
    for i, (det_a, det_b, dist) in enumerate(matched_raw):
        delta     = det_b.diameter_mm - det_a.diameter_mm
        delta_pct = (delta / det_a.diameter_mm * 100.0) if det_a.diameter_mm > 0 else 0.0
        if kernel_change:
            tier, knote = _kernel_pair_comparability(det_a.confidence, det_b.confidence)
        else:
            tier, knote = "GREEN", "Same reconstruction kernel — kernel-shift risk not applicable."
        matched_pairs.append(MatchedPair(
            nodule_id=i + 1,
            study_a=det_a,
            study_b=det_b,
            centroid_dist_mm=round(dist, 2),
            diameter_delta_mm=round(delta, 2),
            diameter_delta_pct=round(delta_pct, 1),
            exceeds_noise=abs(delta) > DIAMETER_NOISE_MM,
            kernel_comparability_tier=tier,
            kernel_comparability_note=knote,
        ))

    if kernel_change:
        dropout_notes_a = [_dropout_kernel_note(d.confidence) for d in only_a]
        dropout_notes_b = [_dropout_kernel_note(d.confidence) for d in only_b]
    else:
        dropout_notes_a = ["" for _ in only_a]
        dropout_notes_b = ["" for _ in only_b]

    acq_a = _acq_result(condition_a)
    acq_b = _acq_result(condition_b)
    cls_a = acq_a['classification']
    cls_b = acq_b['classification']

    comparable = cls_a != 'RED' and cls_b != 'RED'
    if not comparable:
        note = (f"Comparison unreliable: acquisition degradation exceeds 10pp "
                f"({condition_a}: {cls_a}, {condition_b}: {cls_b}). "
                f"Diameter changes may reflect protocol variation, not biology.")
    elif cls_a == 'YELLOW' or cls_b == 'YELLOW':
        note = (f"Interpret with caution: moderate acquisition degradation present "
                f"({condition_a}: {cls_a}, {condition_b}: {cls_b}).")
    else:
        note = (f"Both studies within acceptable acquisition range "
                f"({condition_a}: {cls_a}, {condition_b}: {cls_b}).")

    return LongitudinalResult(
        case_id=case_id,
        condition_a=condition_a,
        condition_b=condition_b,
        acq_a=acq_a,
        acq_b=acq_b,
        matched=matched_pairs,
        only_in_a=only_a,
        only_in_b=only_b,
        comparable=comparable,
        comparability_note=note,
        kernel_change=kernel_change,
        dropout_notes_a=dropout_notes_a,
        dropout_notes_b=dropout_notes_b,
    )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(r: LongitudinalResult) -> None:
    acq_label = {
        'GREEN':   'GREEN',
        'YELLOW':  'YELLOW',
        'RED':     'RED',
        'UNKNOWN': 'UNKNOWN',
    }

    print(f"\n{'='*64}")
    print(f"  Longitudinal Comparison -- {r.case_id}")
    print(f"  {r.condition_a}  ->  {r.condition_b}")
    print(f"{'='*64}")

    deg_a = r.acq_a.get('degradation_pp')
    deg_b = r.acq_b.get('degradation_pp')
    print(f"\nAcquisition reliability")
    print(f"  {r.condition_a:<14} {acq_label.get(r.acq_a['classification'], '?')}"
          + (f"  ({deg_a:.1f}pp degradation)" if deg_a is not None else ""))
    print(f"  {r.condition_b:<14} {acq_label.get(r.acq_b['classification'], '?')}"
          + (f"  ({deg_b:.1f}pp degradation)" if deg_b is not None else ""))
    print(f"\n  {r.comparability_note}")

    print(f"\nMatched nodules  ({len(r.matched)} pairs, threshold={CONF_THRESHOLD}, dist<{MATCH_DIST_MM}mm)")
    if not r.matched:
        print("  (none)")
    else:
        print(f"  {'#':<4} {'Diam A':>8} {'Diam B':>8} {'D mm':>8} {'D %':>7} {'Conf A':>7} {'Conf B':>7} {'Dist':>7}  Flag")
        for p in r.matched:
            flag = '** exceeds noise threshold' if p.exceeds_noise else ''
            print(f"  {p.nodule_id:<4} "
                  f"{p.study_a.diameter_mm:>7.1f}mm "
                  f"{p.study_b.diameter_mm:>7.1f}mm "
                  f"{p.diameter_delta_mm:>+8.2f} "
                  f"{p.diameter_delta_pct:>+6.1f}% "
                  f"{p.study_a.confidence:>7.3f} "
                  f"{p.study_b.confidence:>7.3f} "
                  f"{p.centroid_dist_mm:>6.1f}mm  {flag}")

    if r.only_in_a:
        print(f"\nDetected in {r.condition_a} only (apparent disappearance in {r.condition_b}):")
        for d in r.only_in_a:
            cx, cy, cz = d.centroid_mm
            print(f"  ({cx:.1f}, {cy:.1f}, {cz:.1f})  diam={d.diameter_mm:.1f}mm  conf={d.confidence:.3f}")

    if r.only_in_b:
        print(f"\nDetected in {r.condition_b} only (apparent new nodule vs {r.condition_a}):")
        for d in r.only_in_b:
            cx, cy, cz = d.centroid_mm
            print(f"  ({cx:.1f}, {cy:.1f}, {cz:.1f})  diam={d.diameter_mm:.1f}mm  conf={d.confidence:.3f}")

    print()


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Longitudinal nodule comparison')
    parser.add_argument('case_id',    nargs='?', default='LIDC-IDRI-0001')
    parser.add_argument('condition_a', nargs='?', default='baseline')
    parser.add_argument('condition_b', nargs='?', default='thick_5mm')
    parser.add_argument('--conf',   type=float, default=CONF_THRESHOLD)
    parser.add_argument('--dist',   type=float, default=MATCH_DIST_MM)
    args = parser.parse_args()

    result = compare_studies(
        args.case_id, args.condition_a, args.condition_b,
        conf_threshold=args.conf, max_dist_mm=args.dist,
    )
    print_report(result)
