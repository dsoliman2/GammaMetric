"""
GammaMetric Sensitivity Engine — v1.0
Physics core: maps CT acquisition parameters to estimated AI sensitivity.

Model: MONAI RetinaNet (LUNA16-trained), lung nodule detection.
Baseline derived from 154-case LIDC-IDRI perturbation study (arXiv 2603.26785).

ADDITIVE MODEL LIMITATION: ΔS + ΔK + ΔD is an empirical approximation, not a
physical decomposition. Slice thickness effects are nonlinear; kernel and dose
interact with reconstruction noise texture. Valid only within the characterized
parameter space. Do not extrapolate silently — flag OOD inputs explicitly.
"""

from __future__ import annotations
import logging
import warnings
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Baseline sensitivity (MONAI RetinaNet, LUNA16, v1.0.0)
# ---------------------------------------------------------------------------
BASELINES: dict[str, float] = {
    "overall": 0.848,
    "3-6mm":   0.782,   # critical window — greatest downstream consequence
    "6-10mm":  0.913,
    ">10mm":   0.971,
}

# ---------------------------------------------------------------------------
# Kernel normalization
# ---------------------------------------------------------------------------
# Maps vendor-specific kernel strings → canonical class.
# Keys are lowercase; matching is case-insensitive prefix/exact.
KERNEL_CLASS_MAP: dict[str, str] = {
    # SOFT
    "b40f":    "SOFT",
    "b45f":    "SOFT",
    "b50f":    "SOFT",
    "smooth":  "SOFT",
    "soft":    "SOFT",
    "b":       "SOFT",      # Philips generic soft
    # STANDARD
    "b30f":    "STANDARD",
    "b31f":    "STANDARD",
    "b35f":    "STANDARD",
    "fc10":    "STANDARD",
    "fc11":    "STANDARD",
    "standard": "STANDARD",
    "medium":  "STANDARD",
    "d":       "STANDARD",  # Philips generic medium
    # SHARP
    "b60f":    "SHARP",
    "b70f":    "SHARP",
    "b80f":    "SHARP",
    "lung":    "SHARP",
    "bone":    "SHARP",
    "sharp":   "SHARP",
    "detail":  "SHARP",
    "edge":    "SHARP",
    "fc50":    "SHARP",
    "fc51":    "SHARP",
}

# Per-kernel delta (pp) when the exact kernel string is known.
# Negative = sensitivity loss relative to baseline.
KERNEL_DELTA_MAP: dict[str, float] = {
    "b40f": -7.9,
    "b50f": -10.5,
}

# Class-level fallback deltas when exact kernel isn't characterized.
KERNEL_CLASS_DELTA: dict[str, float] = {
    "STANDARD": 0.0,
    "SOFT":     -7.9,   # conservative: B40f level
    "SHARP":    0.0,    # SHARP is OOD — not characterized; flagged separately
}


def normalize_kernel(kernel_str: str) -> tuple[str, Optional[float], bool]:
    """
    Returns (canonical_class, delta_pp, is_ood).
    delta_pp is None if the class is OOD.
    """
    k = kernel_str.strip().lower().replace("_", "").replace("-", "").replace(" ", "")

    # Exact match first
    if k in KERNEL_CLASS_MAP:
        cls = KERNEL_CLASS_MAP[k]
        delta = KERNEL_DELTA_MAP.get(k, KERNEL_CLASS_DELTA.get(cls))
        ood = cls == "SHARP"
        return cls, (None if ood else delta), ood

    # Prefix match (longest prefix wins — prevents "b" swallowing "bone", etc.)
    for prefix, cls in sorted(KERNEL_CLASS_MAP.items(), key=lambda x: -len(x[0])):
        if k.startswith(prefix):
            delta = KERNEL_CLASS_DELTA.get(cls)
            ood = cls == "SHARP"
            return cls, (None if ood else delta), ood

    return "UNKNOWN", None, True


# ---------------------------------------------------------------------------
# Characterized parameter bounds
# ---------------------------------------------------------------------------
SLICE_THICKNESS_RANGE_MM = (1.25, 5.0)
CTDIVOL_RANGE_MGY        = (2.5, 10.0)
REFERENCE_DOSE_MGY       = 10.0   # dose at which baseline was established

# Characterized operating points for interpolation
# (value, delta_pp) — delta relative to baseline condition
_SLICE_POINTS = [(1.25, 0.0), (2.5, -2.7), (3.75, -8.1), (5.0, -13.2)]
_DOSE_POINTS  = [
    (REFERENCE_DOSE_MGY, 0.0),
    (0.75 * REFERENCE_DOSE_MGY, -1.2),
    (0.50 * REFERENCE_DOSE_MGY, -2.3),
    (0.25 * REFERENCE_DOSE_MGY, -5.1),
]
_DOSE_POINTS.sort(key=lambda x: x[0])  # ascending for interp
_SLICE_POINTS.sort(key=lambda x: x[0])


def _interp(x: float, points: list[tuple[float, float]]) -> float:
    """Piecewise linear interpolation. Clamps at boundaries (no extrapolation)."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    for i in range(len(xs) - 1):
        if xs[i] <= x <= xs[i + 1]:
            t = (x - xs[i]) / (xs[i + 1] - xs[i])
            return ys[i] + t * (ys[i + 1] - ys[i])
    return ys[-1]


def delta_slice(thickness_mm: float) -> float:
    return _interp(thickness_mm, _SLICE_POINTS)


def delta_dose(ctdivol_mgy: float) -> float:
    return _interp(ctdivol_mgy, _DOSE_POINTS)


# ---------------------------------------------------------------------------
# Classification thresholds
# ---------------------------------------------------------------------------
def classify(degradation_pp: float) -> str:
    if degradation_pp < 5.0:
        return "GREEN"
    if degradation_pp <= 10.0:
        return "YELLOW"
    return "RED"


# ---------------------------------------------------------------------------
# Plain language generation
# ---------------------------------------------------------------------------
def _plain_language(
    classification: str,
    degradation_pp: float,
    drivers: list[dict],
    nodule_range: str,
    ood: bool,
) -> str:
    if ood:
        return (
            "One or more acquisition parameters fall outside the characterized envelope. "
            "Sensitivity estimate may not be reliable. Review flagged parameters."
        )
    severity = {"GREEN": "within acceptable limits", "YELLOW": "moderately degraded",
                "RED": "materially degraded"}[classification]
    top = max(drivers, key=lambda d: abs(d.get("contribution_pp", 0)), default=None)
    primary = f" Primary driver is {top['parameter'].replace('_', ' ')}." if top else ""
    return (
        f"Estimated sensitivity for {nodule_range} nodules is {severity} "
        f"under current acquisition conditions ({degradation_pp:.1f}pp loss).{primary}"
    )


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------
@dataclass
class SensitivityInput:
    slice_thickness_mm: float
    reconstruction_kernel: str
    ctdivol_mgy: float
    scanner_model: str = "unknown"
    model_version: str = "unknown"


@dataclass
class SensitivityResult:
    model_version: str
    baseline_sensitivity: float
    estimated_sensitivity: float
    degradation_pp: float
    confidence_interval: list[float]
    classification: str
    nodule_range_most_affected: str
    out_of_distribution: bool
    ood_flags: list[str]
    drivers: list[dict]
    plain_language: str
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "model_version":           self.model_version,
            "baseline_sensitivity":    round(self.baseline_sensitivity, 3),
            "estimated_sensitivity":   round(self.estimated_sensitivity, 3),
            "degradation_pp":          round(self.degradation_pp, 1),
            "confidence_interval":     [round(v, 3) for v in self.confidence_interval],
            "classification":          self.classification,
            "nodule_range_most_affected": self.nodule_range_most_affected,
            "out_of_distribution":     self.out_of_distribution,
            "ood_flags":               self.ood_flags,
            "drivers":                 self.drivers,
            "plain_language":          self.plain_language,
            "warnings":                self.warnings,
        }


def compute(inp: SensitivityInput) -> SensitivityResult:
    """
    Compute sensitivity estimate for a single CT acquisition.
    All degradation deltas are in percentage points (pp).
    """
    warn_msgs: list[str] = []
    ood_flags: list[str] = []

    # model_version guard
    if inp.model_version == "unknown":
        msg = "model_version not provided; historical comparisons will be unreliable."
        logger.warning(msg)
        warn_msgs.append(msg)

    # ── Kernel normalization ────────────────────────────────────────────────
    kernel_class, delta_k, kernel_ood = normalize_kernel(inp.reconstruction_kernel)
    if kernel_ood:
        flag = f"reconstruction_kernel '{inp.reconstruction_kernel}' (class '{kernel_class}') is out of characterized range"
        ood_flags.append(flag)
        logger.warning(flag)
        delta_k = 0.0  # can't estimate; treat as zero but flag OOD

    # ── Slice thickness bounds ──────────────────────────────────────────────
    lo, hi = SLICE_THICKNESS_RANGE_MM
    slice_ood = not (lo <= inp.slice_thickness_mm <= hi)
    if slice_ood:
        flag = f"slice_thickness_mm {inp.slice_thickness_mm} outside characterized range [{lo}, {hi}]"
        ood_flags.append(flag)
        logger.warning(flag)

    # ── Dose bounds ─────────────────────────────────────────────────────────
    lo_d, hi_d = CTDIVOL_RANGE_MGY
    dose_ood = not (lo_d <= inp.ctdivol_mgy <= hi_d)
    if dose_ood:
        flag = f"ctdivol_mgy {inp.ctdivol_mgy} outside characterized range [{lo_d}, {hi_d}]"
        ood_flags.append(flag)
        logger.warning(flag)

    # ── Deltas ─────────────────────────────────────────────────────────────
    delta_s = delta_slice(inp.slice_thickness_mm)
    delta_d = delta_dose(inp.ctdivol_mgy)

    total_delta_pp = delta_s + (delta_k or 0.0) + delta_d

    # Baseline: use 3-6mm range (most affected, clinically critical)
    baseline = BASELINES["3-6mm"]
    estimated = max(0.0, min(1.0, baseline + total_delta_pp / 100.0))
    degradation_pp = abs(total_delta_pp)

    # 95% CI: ±5pp half-width (empirical uncertainty, v1 approximation)
    ci_half = 0.05
    ci = [round(max(0.0, estimated - ci_half), 3),
          round(min(1.0, estimated + ci_half), 3)]

    cls = classify(degradation_pp)
    is_ood = bool(ood_flags)

    # ── Drivers list ────────────────────────────────────────────────────────
    drivers: list[dict] = []
    if abs(delta_s) > 0:
        drivers.append({
            "parameter":       "slice_thickness",
            "value":           f"{inp.slice_thickness_mm}mm",
            "contribution_pp": round(delta_s, 1),
        })
    if abs(delta_k or 0) > 0:
        drivers.append({
            "parameter":        "kernel",
            "value":            inp.reconstruction_kernel,
            "normalized_class": kernel_class,
            "contribution_pp":  round(delta_k, 1),
        })
    if abs(delta_d) > 0:
        drivers.append({
            "parameter":       "dose",
            "value":           f"{inp.ctdivol_mgy}mGy",
            "contribution_pp": round(delta_d, 1),
        })
    # Sort by magnitude descending
    drivers.sort(key=lambda d: abs(d["contribution_pp"]), reverse=True)

    plain = _plain_language(cls, degradation_pp, drivers, "3-6mm", is_ood)

    return SensitivityResult(
        model_version=inp.model_version,
        baseline_sensitivity=baseline,
        estimated_sensitivity=round(estimated, 3),
        degradation_pp=round(degradation_pp, 1),
        confidence_interval=ci,
        classification=cls,
        nodule_range_most_affected="3-6mm",
        out_of_distribution=is_ood,
        ood_flags=ood_flags,
        drivers=drivers,
        plain_language=plain,
        warnings=warn_msgs,
    )
