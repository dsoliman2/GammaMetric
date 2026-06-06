"""GammaMetric ASIR Detector.

Detects iterative reconstruction (GE ASIR, Siemens SAFIRE) from:
  1. GE private DICOM tag (0053,1043) IterativeReconLevel — authoritative when present
  2. Siemens kernel suffix ('f'=FBP, 's'=SAFIRE) — semi-standard
  3. Pixel-based RF classifier trained on QIBA phantom (fallback, DICOM-independent)

Clinical significance (from QIBA phantom study, 140mA):
  - FBP vs ASIR 70%: AUC=0.995 domain separability — AI trained on FBP
    encounters a measurably different input distribution on ASIR images.
  - Apparent lesion diameter differs by 0.29–0.86mm (mean ~0.46mm at
    equatorial slice) — within the range of borderline LI-RADS sizing calls.
  - Effective noise: 32–39% lower with ASIR 70% vs FBP at 140mA, improving
    CNR but shifting the noise texture AI models have learned.
  - ConvolutionKernel tag reads 'STANDARD' for both FBP and all ASIR levels —
    standard metadata pipelines cannot detect this condition.
"""
from __future__ import annotations

import os
import dataclasses as dc
from typing import Optional

import numpy as np
from scipy.ndimage import sobel

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'asir_detector.joblib')

# GE private tag for iterative reconstruction level
GE_PRIVATE_TAG = (0x0053, 0x1043)   # IterativeReconLevel

# Siemens kernel suffix: 'f' = FBP, 's' = SAFIRE
SIEMENS_SAFIRE_SUFFIX = 's'

# Pixel sampling parameters (must match training)
_PATCH    = 32
_N_PATCH  = 100
_N_SLICE  = 10
_HU_MIN   = -200   # body tissue (not air)
_HU_MAX   =  200
_STD_MAX  =  80


@dc.dataclass
class ASIRResult:
    detected: bool                   # True if iterative recon found
    level: Optional[str]             # '30%' / '50%' / '70%' / 'SAFIRE' / None
    confidence: float                # 0–1; 1.0 if from authoritative tag
    method: str                      # 'private_tag' | 'kernel_suffix' | 'pixel' | 'none'
    private_tag_value: Optional[str] # raw value from (0053,1043) if present

    @property
    def metadata_captured(self) -> bool:
        """True if the standard ConvolutionKernel tag would capture this."""
        return self.method == 'kernel_suffix'  # only Siemens 'f'/'s' is in the tag

    def alert_line(self) -> str:
        if not self.detected:
            return ""
        tag_note = ("not captured by standard ConvolutionKernel tag"
                    if not self.metadata_captured
                    else "encoded in kernel suffix only (no standardised field)")
        level_str = f" ({self.level})" if self.level else ""
        return (
            f"Iterative reconstruction detected{level_str} via {self.method} "
            f"[{tag_note}]. Apparent lesion size may differ from FBP baseline "
            f"by ~0.3–0.9 mm; input distribution shift AUC ≈ 0.99 vs FBP."
        )


# ---------------------------------------------------------------------------
# Tag-based detection (fast, no pixel access needed)
# ---------------------------------------------------------------------------

def _check_private_tag(dicom_ds) -> ASIRResult:
    """Check GE private tag (0053,1043) for ASIR level."""
    try:
        val = dicom_ds[GE_PRIVATE_TAG].value
        val_str = str(val).strip()
        if val_str and val_str not in ('0', '', 'None'):
            level = val_str if '%' in val_str else f"{val_str}%"
            return ASIRResult(detected=True, level=level, confidence=1.0,
                              method='private_tag', private_tag_value=val_str)
    except (KeyError, Exception):
        pass
    return ASIRResult(detected=False, level=None, confidence=1.0,
                      method='none', private_tag_value=None)


def _check_kernel_suffix(kernel: str) -> ASIRResult:
    """Detect Siemens SAFIRE from kernel suffix (e.g. B20s vs B20f)."""
    if kernel and kernel[-1].lower() == SIEMENS_SAFIRE_SUFFIX:
        return ASIRResult(detected=True, level='SAFIRE', confidence=0.95,
                          method='kernel_suffix', private_tag_value=None)
    return ASIRResult(detected=False, level=None, confidence=0.95,
                      method='none', private_tag_value=None)


def check_dicom_tags(dicom_ds, kernel: Optional[str] = None) -> ASIRResult:
    """
    Check DICOM tags for iterative reconstruction. Fast — no pixel access.
    Tries GE private tag first, then Siemens kernel suffix.
    """
    result = _check_private_tag(dicom_ds)
    if result.detected:
        return result
    k = kernel or str(dicom_ds.get('ConvolutionKernel', '') or '')
    return _check_kernel_suffix(k)


# ---------------------------------------------------------------------------
# Pixel-based detection (fallback when only standard tags available)
# ---------------------------------------------------------------------------

def _patch_features(patch: np.ndarray) -> list[float]:
    ns = float(patch.std())
    gx = sobel(patch, axis=0); gy = sobel(patch, axis=1)
    sh = float(np.sqrt(gx**2 + gy**2).mean())
    ft = np.abs(np.fft.fft2(patch)); H = patch.shape[0] // 2
    fr = float((ft[H//2:, H//2:].mean() + 1e-9) / (ft[:H//2, :H//2].mean() + 1e-9))
    flat = patch.flatten()
    cr = float(np.corrcoef(flat[:-1], flat[1:])[0, 1]) if len(flat) > 1 else 0.0
    return [ns, sh, fr, cr]


def _sample_patches(slices: list[np.ndarray]) -> np.ndarray:
    rng = np.random.default_rng(42)
    rows = []
    for sl in slices:
        H, W = sl.shape
        if H < _PATCH or W < _PATCH:
            continue
        found = attempts = 0
        while found < _N_PATCH and attempts < _N_PATCH * 15:
            r = rng.integers(0, H - _PATCH)
            c = rng.integers(0, W - _PATCH)
            patch = sl[r:r + _PATCH, c:c + _PATCH]
            if _HU_MIN < patch.mean() < _HU_MAX and patch.std() < _STD_MAX:
                rows.append(_patch_features(patch))
                found += 1
            attempts += 1
    return np.array(rows) if rows else np.zeros((1, 4))


class ASIRDetector:
    """Pixel-based ASIR detector. Load once, call detect() per volume."""

    def __init__(self, model_path: str = MODEL_PATH):
        import joblib
        bundle = joblib.load(model_path)
        self._model   = bundle['model']
        self._scaler  = bundle['scaler']
        self._cv_auc  = bundle.get('cv_auc_mean', 0.0)

    def detect(self, slices: list[np.ndarray]) -> ASIRResult:
        """
        slices: list of 2D HU arrays (central slices of the volume).
        Returns ASIRResult with method='pixel'.
        """
        mid = len(slices) // 2
        sel = slices[max(0, mid - _N_SLICE // 2): mid + _N_SLICE // 2]
        feats = _sample_patches(sel)
        if len(feats) < 10:
            return ASIRResult(detected=False, level=None, confidence=0.5,
                              method='pixel', private_tag_value=None)
        Xs = self._scaler.transform(feats)
        probs = self._model.predict_proba(Xs)[:, 1]
        mean_prob = float(probs.mean())
        detected = mean_prob >= 0.5
        return ASIRResult(
            detected=detected,
            level=None,   # binary classifier; level needs multi-class extension
            confidence=round(mean_prob if detected else 1 - mean_prob, 3),
            method='pixel',
            private_tag_value=None,
        )


# ---------------------------------------------------------------------------
# Convenience: detect from DICOM directory (tags first, pixels as fallback)
# ---------------------------------------------------------------------------

def detect_from_dicom_dir(dicom_dir: str,
                           kernel: Optional[str] = None) -> ASIRResult:
    """
    Full detection pipeline for a DICOM series directory.
    1. Read one DICOM file, check tags.
    2. If inconclusive, run pixel classifier.
    """
    import pydicom

    representative = None
    slices = []
    for fn in sorted(os.listdir(dicom_dir)):
        p = os.path.join(dicom_dir, fn)
        try:
            ds = pydicom.dcmread(p, stop_before_pixels=False)
            if representative is None:
                representative = ds
            arr = ds.pixel_array.astype(np.float32)
            arr = arr * float(getattr(ds, 'RescaleSlope', 1)) + float(getattr(ds, 'RescaleIntercept', 0))
            slices.append(arr)
        except Exception:
            continue

    if representative is not None:
        tag_result = check_dicom_tags(representative, kernel)
        if tag_result.detected or tag_result.method != 'none':
            return tag_result

    if not os.path.exists(MODEL_PATH):
        return ASIRResult(detected=False, level=None, confidence=0.0,
                          method='pixel', private_tag_value=None)

    detector = ASIRDetector(MODEL_PATH)
    return detector.detect(slices)
