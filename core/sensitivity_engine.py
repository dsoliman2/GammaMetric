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
import math
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
# Diameter uncertainty tables  (empirical; 3150 matched LIDC-IDRI nodule pairs)
# Tuple: (mean_shift_mm, uncertainty_95ci_mm)  — positive = overestimation
# Slice table: how much the AI over/underestimates diameter at each slice thickness.
# Dose table: additional bias at reduced dose (keyed by ctdivol/reference fraction).
# ---------------------------------------------------------------------------
_DIAM_UNC_SLICE: dict[float, dict[str, tuple[float, float]]] = {
    1.25: {"overall": (0.0, 0.0)},
    3.0: {
        "overall": (0.30, 0.95),
        "3-6mm":   (0.17, 0.71),
        "6-10mm":  (0.25, 1.66),
        "10-20mm": (0.49, 1.07),
        "20-50mm": (1.12, 10.10),
    },
    5.0: {
        "overall": (1.70, 9.88),
        "3-6mm":   (0.52, 1.52),
        "6-10mm":  (1.57, 8.55),
        "10-20mm": (1.84, 10.69),
        "20-50mm": (7.85, 33.26),
    },
}

# Kernel class → size stratum → (mean_shift_mm, uncertainty_95ci_mm)
# SOFT kernel smooths edges: underestimates large nodules, slight over on small.
_DIAM_UNC_KERNEL: dict[str, dict[str, tuple[float, float]]] = {
    "SOFT": {
        "overall": (-0.12, 4.35),
        "3-6mm":   ( 0.41, 1.18),
        "6-10mm":  (-0.12, 3.58),
        "10-20mm": (-0.13, 6.11),
        "20-50mm": (-2.62, 14.86),
    },
    # Sharp kernel tightens bounding boxes via edge enhancement — consistent underestimation.
    "SHARP": {
        "overall": (-0.70, 3.79),
        "3-6mm":   (-0.12, 0.72),
        "6-10mm":  (-0.81, 3.68),
        "10-20mm": (-1.25, 5.42),
        "20-50mm": (-3.98, 15.18),
    },
    # STANDARD: no kernel-driven diameter bias (baseline condition)
    "STANDARD": {
        "overall": (0.0, 0.0),
    },
}

_DIAM_UNC_DOSE: dict[float, dict[str, tuple[float, float]]] = {
    0.75: {  # dose_25pct (7.5 mGy at 10 mGy reference)
        "overall": (0.19, 2.34),
        "3-6mm":   (0.00, 1.22),
        "6-10mm":  (0.08, 1.71),
        "10-20mm": (0.48, 2.76),
        "20-50mm": (1.06, 6.41),
    },
    0.50: {  # dose_50pct (5.0 mGy)
        "overall": (-0.02, 1.04),
        "3-6mm":   (-0.02, 0.83),
        "6-10mm":  ( 0.02, 0.87),
        "10-20mm": (-0.03, 1.23),
        "20-50mm": (-0.28, 1.77),
    },
}


def _size_key(diameter_mm: Optional[float]) -> str:
    if diameter_mm is None:
        return "overall"
    if diameter_mm < 6:
        return "3-6mm"
    if diameter_mm < 10:
        return "6-10mm"
    if diameter_mm < 20:
        return "10-20mm"
    return "20-50mm"


def _interp_diam(x: float, table: dict, key: str) -> tuple[float, float]:
    xs = sorted(table.keys())

    def _get(k: float) -> tuple[float, float]:
        d = table[k]
        return d.get(key, d["overall"])

    if x <= xs[0]:
        return _get(xs[0])
    if x >= xs[-1]:
        return _get(xs[-1])
    for i in range(len(xs) - 1):
        lo, hi = xs[i], xs[i + 1]
        if lo <= x <= hi:
            t = (x - lo) / (hi - lo)
            m0, c0 = _get(lo)
            m1, c1 = _get(hi)
            return m0 + t * (m1 - m0), c0 + t * (c1 - c0)
    return _get(xs[-1])


def compute_diameter_uncertainty(
    slice_thickness_mm: float,
    ctdivol_mgy: float,
    nodule_diameter_mm: Optional[float] = None,
    kernel_class: str = "STANDARD",
) -> dict:
    """
    Empirical AI diameter measurement uncertainty for given CT acquisition parameters.

    Returns mean expected shift (positive = overestimation), 95% CI width, size stratum,
    and dominant driver. Based on 3799 matched LIDC-IDRI pairs (arXiv 2603.26785).
    """
    key = _size_key(nodule_diameter_mm)

    # Slice contribution
    s_mean, s_ci = _interp_diam(slice_thickness_mm, _DIAM_UNC_SLICE, key)

    # Dose contribution
    dose_frac = ctdivol_mgy / REFERENCE_DOSE_MGY
    dose_xs   = sorted(_DIAM_UNC_DOSE.keys())   # [0.50, 0.75]
    d_mean = d_ci = 0.0
    if dose_frac < 1.0:
        if dose_frac <= dose_xs[0]:
            e = _DIAM_UNC_DOSE[dose_xs[0]]
            d_mean, d_ci = e.get(key, e["overall"])
        elif dose_frac >= dose_xs[-1]:
            t = (dose_frac - dose_xs[-1]) / (1.0 - dose_xs[-1])
            e = _DIAM_UNC_DOSE[dose_xs[-1]]
            m0, c0 = e.get(key, e["overall"])
            d_mean, d_ci = m0 * (1 - t), c0 * (1 - t)
        else:
            for i in range(len(dose_xs) - 1):
                lo, hi = dose_xs[i], dose_xs[i + 1]
                if lo <= dose_frac <= hi:
                    t = (dose_frac - lo) / (hi - lo)
                    le = _DIAM_UNC_DOSE[lo]
                    he = _DIAM_UNC_DOSE[hi]
                    m0, c0 = le.get(key, le["overall"])
                    m1, c1 = he.get(key, he["overall"])
                    d_mean, d_ci = m0 + t * (m1 - m0), c0 + t * (c1 - c0)
                    break

    # Kernel contribution
    k_mean = k_ci = 0.0
    kern_entry = _DIAM_UNC_KERNEL.get(kernel_class.upper())
    if kern_entry:
        k_mean, k_ci = kern_entry.get(key, kern_entry["overall"])

    combined_ci   = round(math.sqrt(s_ci ** 2 + d_ci ** 2 + k_ci ** 2), 2)
    combined_mean = round(s_mean + d_mean + k_mean, 2)

    # Dominant driver by individual CI contribution
    contribs = {"slice_thickness": s_ci, "dose": d_ci, "kernel": k_ci}
    dominant = max(contribs, key=contribs.get)

    return {
        "mean_shift_mm":       combined_mean,
        "uncertainty_95ci_mm": combined_ci,
        "size_stratum":        key,
        "dominant_driver":     dominant,
    }


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
    diameter_uncertainty: Optional[dict] = field(default=None)

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
            "diameter_uncertainty":    self.diameter_uncertainty,
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
    diam_unc = compute_diameter_uncertainty(
        inp.slice_thickness_mm, inp.ctdivol_mgy, kernel_class=kernel_class
    )

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
        diameter_uncertainty=diam_unc,
    )
