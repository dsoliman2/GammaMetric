"""GammaMetric Acquisition Fingerprint — DICOM-header-independent classifier.

Predicts the acquisition condition of a CT volume from pixel statistics alone,
without trusting (or requiring) DICOM metadata. Designed to be:

  - A reliability layer for AI deployment pipelines: verifies the input scan
    matches the conditions the AI was validated on, even when DICOM headers
    are missing, anonymized away, or wrong.
  - A drift-detection signal for prospective monitoring: a fingerprint vector
    per scan that can be tracked across time at a deployed site.
  - A foundation for per-lesion vulnerability prediction (downstream).

The current classifier is a Random Forest over 4 patch-level features
(noise_std, sharpness, freq_ratio, corr_ratio) trained on n=21 LIDC-IDRI cases
across 7 simulated acquisition conditions, aggregated to volume-level
predictions by majority vote over patches.

Train once with `train_and_save`, then load and use:

    fp = AcquisitionFingerprint.load(MODEL_PATH)
    result = fp.predict('path/to/scan.nii.gz')
    # result.condition, result.probabilities, result.confidence

DICOM header check:

    check = fp.check_against_dicom('path/to/scan.nii.gz', 'path/to/dicom_dir')
    # check.match, check.claimed_condition, check.predicted_condition

Honest performance caveat: LOCO accuracy is 59% across all 7 conditions on the
n=21 training set, with thick-slice conditions at chance (the simulated slice-
thickness resampling does not leave a recognizable pixel-statistic signature).
The classifier is reliable on dose and kernel conditions, less so on thickness.
"""
from __future__ import annotations

import os
import json
import glob
import dataclasses as dc
from typing import Optional

import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.ndimage import sobel
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

CONDITIONS = ['baseline', 'dose_25pct', 'dose_50pct',
              'thick_3mm', 'thick_5mm',
              'soft_kernel', 'sharp_kernel']

DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'models', 'acquisition_fingerprint_rf.joblib'
)

# ---------------------------------------------------------------------------
# Feature extraction (must match training-time exactly)
# ---------------------------------------------------------------------------

PATCH = 64               # 2D patch size for parenchyma sampling
LUNG_HU_MIN = -950       # parenchyma window
LUNG_HU_MAX = -300
MIN_LUNG_FRAC = 0.7      # require >=70% lung-attenuation voxels in patch
PATCHES_PER_SLICE_MAX = 8
SLICES_PER_VOLUME = 12


def _extract_patch_features(patch: np.ndarray) -> tuple[float, float, float, float]:
    """Four pixel features per 2D parenchyma patch."""
    noise_std = float(patch.std())
    gx = sobel(patch, axis=0)
    gy = sobel(patch, axis=1)
    sharpness = float(np.sqrt(gx * gx + gy * gy).mean())
    fft = np.fft.fftshift(np.fft.fft2(patch - patch.mean()))
    power = np.abs(fft) ** 2
    n = patch.shape[0]
    fy, fx = np.mgrid[0:n, 0:n]
    fr = np.sqrt((fy - n // 2) ** 2 + (fx - n // 2) ** 2)
    low = float(power[fr < n / 6].mean()) + 1e-12
    high = float(power[fr > n / 3].mean()) + 1e-12
    freq_ratio = high / low
    # Voxel-to-voxel autocorrelation ratio (lag-1 horizontal / lag-2 horizontal)
    c1 = float(np.mean((patch[:, :-1] - patch.mean()) * (patch[:, 1:] - patch.mean())))
    c2 = float(np.mean((patch[:, :-2] - patch.mean()) * (patch[:, 2:] - patch.mean())))
    corr_ratio = c1 / (c2 + 1e-9)
    return noise_std, sharpness, freq_ratio, corr_ratio


def _extract_volume_features(volume: np.ndarray, seed: int = 0) -> np.ndarray:
    """Extract per-patch features from a 3D volume. Returns (n_patches, 4) array."""
    rng = np.random.default_rng(seed)
    z_dim = volume.shape[0]
    if z_dim < 5:
        return np.zeros((0, 4))
    z_idx = rng.choice(z_dim, size=min(SLICES_PER_VOLUME, z_dim), replace=False)
    half = PATCH // 2
    feats = []
    for z in z_idx:
        slc = volume[z]
        in_lung = (slc >= LUNG_HU_MIN) & (slc <= LUNG_HU_MAX)
        H, W = slc.shape
        cands = []
        # Sliding grid of candidate patch centers
        for y in range(half, H - half, PATCH // 2):
            for x in range(half, W - half, PATCH // 2):
                roi = in_lung[y - half:y + half, x - half:x + half]
                if roi.mean() >= MIN_LUNG_FRAC:
                    cands.append((y, x))
        rng.shuffle(cands)
        for y, x in cands[:PATCHES_PER_SLICE_MAX]:
            patch = slc[y - half:y + half, x - half:x + half].astype(np.float32)
            feats.append(_extract_patch_features(patch))
    return np.asarray(feats, dtype=np.float32)


def load_volume(path: str) -> np.ndarray:
    """Load CT volume from NIfTI or DICOM dir; return (Z, Y, X) HU array."""
    if os.path.isdir(path):
        reader = sitk.ImageSeriesReader()
        files = reader.GetGDCMSeriesFileNames(path)
        if not files:
            raise ValueError(f"No DICOM series found in {path}")
        reader.SetFileNames(files)
        img = reader.Execute()
    else:
        img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img).astype(np.float32)
    return arr


# ---------------------------------------------------------------------------
# DICOM header → condition coarse mapping
# ---------------------------------------------------------------------------

SOFT_KERNEL_TOKENS = {'B20F', 'B30F', 'B40F', 'STANDARD', 'SOFT', 'C', 'FC10', 'FC11'}
SHARP_KERNEL_TOKENS = {'B70F', 'B75F', 'B80F', 'LUNG', 'BONE', 'D', 'FC51', 'FC52'}


def claimed_condition_from_dicom(dicom_dir: str) -> dict:
    """Read DICOM header and infer which model class best matches the claim.
    Returns dict with metadata + claimed_condition (one of CONDITIONS or 'unknown')."""
    try:
        import pydicom
    except ImportError:
        return {'error': 'pydicom not installed'}
    dcm_files = list(glob.iglob(os.path.join(dicom_dir, '**', '*.dcm'), recursive=True))
    if not dcm_files:
        return {'error': f'no .dcm files under {dicom_dir}'}
    ds = pydicom.dcmread(dcm_files[0], stop_before_pixels=True)
    mfg = str(getattr(ds, 'Manufacturer', '')).strip().upper()
    kernel = str(getattr(ds, 'ConvolutionKernel', '')).strip()
    slice_mm = float(getattr(ds, 'SliceThickness', 0.0) or 0.0)
    ctdivol = float(getattr(ds, 'CTDIvol', 0.0) or 0.0)
    k_upper = kernel.upper().replace(' ', '')
    claimed = 'baseline'
    if any(t in k_upper for t in SHARP_KERNEL_TOKENS):
        claimed = 'sharp_kernel'
    elif any(t in k_upper for t in SOFT_KERNEL_TOKENS):
        claimed = 'soft_kernel'
    if slice_mm >= 4.5:
        claimed = 'thick_5mm'
    elif slice_mm >= 2.7:
        claimed = 'thick_3mm'
    return {
        'manufacturer': mfg,
        'kernel': kernel,
        'slice_thickness_mm': slice_mm,
        'ctdivol_mgy': ctdivol,
        'claimed_condition': claimed,
    }


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dc.dataclass
class PredictResult:
    image: str
    n_patches: int
    predicted_condition: str
    confidence: float
    probabilities: dict
    feature_means: dict

    def to_dict(self) -> dict:
        return dc.asdict(self)


@dc.dataclass
class HeaderCheckResult:
    image: str
    dicom_dir: str
    claimed_condition: str
    predicted_condition: str
    match: bool
    predicted_confidence: float
    header: dict
    probabilities: dict

    def to_dict(self) -> dict:
        return dc.asdict(self)


# ---------------------------------------------------------------------------
# Core API
# ---------------------------------------------------------------------------

class AcquisitionFingerprint:
    """DICOM-header-independent acquisition-condition classifier."""

    def __init__(self, model, scaler, classes, feature_names):
        self.model = model
        self.scaler = scaler
        self.classes = list(classes)
        self.feature_names = list(feature_names)

    # ── Persistence ──────────────────────────────────────────────────────────
    def save(self, path: str = DEFAULT_MODEL_PATH) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'classes': self.classes,
            'feature_names': self.feature_names,
        }, path)

    @classmethod
    def load(cls, path: str = DEFAULT_MODEL_PATH) -> 'AcquisitionFingerprint':
        d = joblib.load(path)
        return cls(d['model'], d['scaler'], d['classes'], d['feature_names'])

    # ── Training (one-time) ──────────────────────────────────────────────────
    @classmethod
    def train_from_csv(cls, csv_path: str, n_estimators: int = 400,
                       max_depth: int = 10, random_state: int = 42
                       ) -> 'AcquisitionFingerprint':
        df = pd.read_csv(csv_path)
        feature_names = ['noise_std', 'sharpness', 'freq_ratio', 'corr_ratio']
        X = df[feature_names].values.astype(np.float32)
        y = df['condition'].astype(str).values
        scaler = StandardScaler().fit(X)
        Xs = scaler.transform(X)
        rf = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            n_jobs=-1, random_state=random_state, class_weight='balanced',
        )
        rf.fit(Xs, y)
        return cls(rf, scaler, rf.classes_, feature_names)

    # ── Inference ────────────────────────────────────────────────────────────
    def _predict_patches(self, X_patches: np.ndarray) -> tuple[str, float, dict, dict]:
        if X_patches.shape[0] == 0:
            return ('unknown', 0.0,
                    {c: 0.0 for c in self.classes},
                    {f: float('nan') for f in self.feature_names})
        Xs = self.scaler.transform(X_patches)
        probs = self.model.predict_proba(Xs)  # (n_patches, n_classes)
        mean_probs = probs.mean(axis=0)
        cond_idx = int(np.argmax(mean_probs))
        condition = self.classes[cond_idx]
        confidence = float(mean_probs[cond_idx])
        prob_dict = {self.classes[i]: float(mean_probs[i]) for i in range(len(self.classes))}
        feat_means = {self.feature_names[i]: float(X_patches[:, i].mean())
                      for i in range(len(self.feature_names))}
        return condition, confidence, prob_dict, feat_means

    def predict(self, image_path: str, seed: int = 0) -> PredictResult:
        """Predict acquisition condition from a NIfTI file or DICOM directory."""
        volume = load_volume(image_path)
        X = _extract_volume_features(volume, seed=seed)
        condition, confidence, probs, feat_means = self._predict_patches(X)
        return PredictResult(
            image=image_path,
            n_patches=int(X.shape[0]),
            predicted_condition=condition,
            confidence=confidence,
            probabilities=probs,
            feature_means=feat_means,
        )

    def check_against_dicom(self, image_path: str, dicom_dir: str,
                            seed: int = 0) -> HeaderCheckResult:
        """Run prediction + DICOM-header lookup; report whether they match."""
        pred = self.predict(image_path, seed=seed)
        header = claimed_condition_from_dicom(dicom_dir)
        claimed = header.get('claimed_condition', 'unknown')
        match = (claimed == pred.predicted_condition)
        return HeaderCheckResult(
            image=image_path,
            dicom_dir=dicom_dir,
            claimed_condition=claimed,
            predicted_condition=pred.predicted_condition,
            match=bool(match),
            predicted_confidence=pred.confidence,
            header=header,
            probabilities=pred.probabilities,
        )


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def _emit_json(obj, output_path: Optional[str]) -> None:
    txt = json.dumps(obj, indent=2)
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(txt + '\n')
    else:
        print(txt)


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description='GammaMetric Acquisition Fingerprint CLI')
    sub = p.add_subparsers(dest='cmd', required=True)

    p_train = sub.add_parser('train', help='Train and save the classifier from features CSV')
    p_train.add_argument('--csv', required=True, help='features CSV with columns: case_id,condition,noise_std,sharpness,freq_ratio,corr_ratio')
    p_train.add_argument('--model', default=DEFAULT_MODEL_PATH)

    p_pred = sub.add_parser('predict', help='Predict acquisition condition for an image')
    p_pred.add_argument('--image', required=True, help='NIfTI path or DICOM directory')
    p_pred.add_argument('--model', default=DEFAULT_MODEL_PATH)
    p_pred.add_argument('--output', default=None, help='Write JSON report here (default: stdout)')

    p_check = sub.add_parser('check', help='Predict + compare to DICOM header')
    p_check.add_argument('--image', required=True)
    p_check.add_argument('--dicom', required=True, help='DICOM directory with the original headers')
    p_check.add_argument('--model', default=DEFAULT_MODEL_PATH)
    p_check.add_argument('--output', default=None)

    args = p.parse_args()
    if args.cmd == 'train':
        fp = AcquisitionFingerprint.train_from_csv(args.csv)
        fp.save(args.model)
        print(f"Model saved to {args.model}")
        print(f"Classes: {fp.classes}")
        # Quick LOCO evaluation
        df = pd.read_csv(args.csv)
        from sklearn.model_selection import LeaveOneGroupOut
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score
        X = df[fp.feature_names].values.astype(np.float32)
        y = df['condition'].astype(str).values
        groups = df['case_id'].astype(str).values
        Xs = fp.scaler.transform(X)
        logo = LeaveOneGroupOut()
        preds = np.empty(len(y), dtype=object)
        for tr, te in logo.split(Xs, y, groups):
            m = RandomForestClassifier(n_estimators=400, max_depth=10,
                                       n_jobs=-1, random_state=42,
                                       class_weight='balanced').fit(Xs[tr], y[tr])
            preds[te] = m.predict(Xs[te])
        acc = accuracy_score(y, preds)
        print(f"LOCO accuracy across {df['case_id'].nunique()} cases: {acc:.1%}")
        print(f"Per-condition LOCO accuracy:")
        for c in sorted(set(y)):
            m = y == c
            if m.sum() == 0: continue
            print(f"  {c:<14}  {accuracy_score(y[m], preds[m]):.1%}  (n={int(m.sum())})")
        return
    if args.cmd == 'predict':
        fp = AcquisitionFingerprint.load(args.model)
        result = fp.predict(args.image)
        _emit_json(result.to_dict(), args.output)
        return
    if args.cmd == 'check':
        fp = AcquisitionFingerprint.load(args.model)
        result = fp.check_against_dicom(args.image, args.dicom)
        _emit_json(result.to_dict(), args.output)
        return


if __name__ == '__main__':
    main()
