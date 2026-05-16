"""
Pixel-based acquisition condition classifier.

Tests whether degradation condition can be predicted from image pixels alone
(no DICOM metadata) using three lung-parenchyma patch features:
  1. Noise magnitude  — std of HU values in patch
  2. Edge sharpness   — mean Sobel gradient magnitude
  3. Frequency ratio  — high-freq / low-freq power from 2D FFT

Multiclass logistic regression across 6 conditions.
Chance = ~17%. Target > 50%.

Usage:
  python pixel_experiment.py              # full run: generate + extract + classify
  python pixel_experiment.py --skip-gen   # skip NIfTI generation (already done)
  python pixel_experiment.py --skip-gen --skip-extract  # classify only from CSV
"""

import argparse
import os
import glob
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.ndimage import sobel, gaussian_filter
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DICOM_BASE  = r'C:\Users\Dan\Desktop\gammametric_output\Lung DICOM\manifest-1600709154662\LIDC-IDRI'
NIFTI_OUT   = r'G:\GammaMetric\pixel_niftis'
FEATURES_CSV = r'G:\GammaMetric\pixel_niftis\features.csv'
RANDOM_SEED = 42

CONDITIONS = ['baseline', 'dose_25pct', 'dose_50pct', 'thick_3mm', 'thick_5mm', 'soft_kernel']

CASES = [
    'LIDC-IDRI-0001', 'LIDC-IDRI-0002', 'LIDC-IDRI-0003', 'LIDC-IDRI-0004',
    'LIDC-IDRI-0005', 'LIDC-IDRI-0006', 'LIDC-IDRI-0007', 'LIDC-IDRI-0008',
    'LIDC-IDRI-0009', 'LIDC-IDRI-0010', 'LIDC-IDRI-0011', 'LIDC-IDRI-0012',
    'LIDC-IDRI-0016', 'LIDC-IDRI-0042', 'LIDC-IDRI-0043', 'LIDC-IDRI-0086',
    'LIDC-IDRI-0087', 'LIDC-IDRI-0089', 'LIDC-IDRI-0092', 'LIDC-IDRI-0093',
    'LIDC-IDRI-0151',
]

# LIDC-0009 already has NIfTIs on the Desktop — reuse them
LEGACY_NIFTIS = {
    'LIDC-IDRI-0009': {
        'baseline':   r'C:\Users\Dan\Desktop\gammametric_output\LIDC-0009.nii.gz',
        'dose_25pct': r'C:\Users\Dan\Desktop\gammametric_output\LIDC-0009_dose_25pct.nii.gz',
        'dose_50pct': r'C:\Users\Dan\Desktop\gammametric_output\LIDC-0009_dose_50pct.nii.gz',
        'thick_3mm':  r'C:\Users\Dan\Desktop\gammametric_output\LIDC-0009_thick_3mm.nii.gz',
        'thick_5mm':  r'C:\Users\Dan\Desktop\gammametric_output\LIDC-0009_thick_5mm.nii.gz',
    }
}


# ---------------------------------------------------------------------------
# NIfTI generation (same logic as batch_pipeline_v2.py)
# ---------------------------------------------------------------------------

def find_largest_ct_series(case_id):
    case_dir = os.path.join(DICOM_BASE, case_id)
    if not os.path.exists(case_dir):
        return None
    best_dir, best_count = None, 0
    for dirpath, _, files in os.walk(case_dir):
        dcms = [f for f in files if f.lower().endswith('.dcm')]
        if len(dcms) > best_count:
            best_count = len(dcms)
            best_dir = dirpath
    return best_dir


def dicom_to_sitk(dicom_dir):
    reader = sitk.ImageSeriesReader()
    files = reader.GetGDCMSeriesFileNames(dicom_dir)
    reader.SetFileNames(files)
    return reader.Execute()


def apply_dose_reduction(image, dose_fraction, rng):
    arr = sitk.GetArrayFromImage(image).astype(np.float32)
    tissue_mask = (arr > -500) & (arr < 500)
    sigma_original = np.std(arr[tissue_mask]) if tissue_mask.any() else 20.0
    sigma_added = sigma_original * np.sqrt((1.0 - dose_fraction) / dose_fraction)
    noise = rng.normal(0, sigma_added, arr.shape)
    noisy = np.clip(arr + noise, -1024, 3071).astype(np.int16)
    out = sitk.GetImageFromArray(noisy)
    out.CopyInformation(image)
    return out


def apply_thick_slices(image, target_mm):
    spacing = list(image.GetSpacing())
    if spacing[2] >= target_mm:
        return image
    factor = max(1, int(round(target_mm / spacing[2])))
    arr = sitk.GetArrayFromImage(image).astype(np.float32)
    slabs = [arr[i:i+factor].mean(axis=0)
             for i in range(0, arr.shape[0] - factor + 1, factor)]
    if not slabs:
        return image
    thick_arr = np.stack(slabs).astype(np.int16)
    out = sitk.GetImageFromArray(thick_arr)
    out.SetSpacing((spacing[0], spacing[1], spacing[2] * factor))
    out.SetOrigin(image.GetOrigin())
    out.SetDirection(image.GetDirection())
    return out


def apply_soft_kernel(image, sigma_mm=1.5):
    """
    Simulate soft reconstruction kernel via in-plane Gaussian smoothing.
    sigma_mm is physical mm; converted to voxels using xy spacing.
    Only blurs in-plane (axes 1,2 in ZYX) — slice direction unchanged.
    """
    spacing = image.GetSpacing()  # (x, y, z) in mm
    sigma_x = sigma_mm / spacing[0]
    sigma_y = sigma_mm / spacing[1]
    arr = sitk.GetArrayFromImage(image).astype(np.float32)
    smoothed = gaussian_filter(arr, sigma=(0, sigma_y, sigma_x))
    out = sitk.GetImageFromArray(smoothed.astype(np.int16))
    out.CopyInformation(image)
    return out


def generate_niftis():
    rng = np.random.default_rng(RANDOM_SEED)
    os.makedirs(NIFTI_OUT, exist_ok=True)

    for case_id in CASES:
        print(f'\n{case_id}')

        # For LIDC-0009: use legacy NIfTIs for existing conditions,
        # but generate soft_kernel from the legacy baseline
        if case_id in LEGACY_NIFTIS:
            soft_path = os.path.join(NIFTI_OUT, f'{case_id}_soft_kernel.nii.gz')
            if os.path.exists(soft_path):
                print(f'  [skip] using legacy NIfTIs (soft_kernel exists)')
            else:
                baseline_path = LEGACY_NIFTIS[case_id]['baseline']
                print(f'  generating soft_kernel from legacy baseline...', end=' ', flush=True)
                try:
                    img = sitk.ReadImage(baseline_path)
                    sitk.WriteImage(apply_soft_kernel(img), soft_path)
                    print(f'done ({os.path.getsize(soft_path)/1e6:.1f} MB)')
                except Exception as e:
                    print(f'ERROR: {e}')
            continue

        # Check if all 5 conditions already generated
        paths = {c: os.path.join(NIFTI_OUT, f'{case_id}_{c}.nii.gz') for c in CONDITIONS}
        if all(os.path.exists(p) for p in paths.values()):
            print(f'  [skip] all conditions exist')
            continue

        dicom_dir = find_largest_ct_series(case_id)
        if not dicom_dir:
            print(f'  [skip] DICOM not found')
            continue

        print(f'  loading DICOM from {dicom_dir}...', end=' ', flush=True)
        try:
            image = dicom_to_sitk(dicom_dir)
            print(f'loaded {image.GetSize()} spacing={[round(s,2) for s in image.GetSpacing()]}')
        except Exception as e:
            print(f'ERROR: {e}')
            continue

        for condition in CONDITIONS:
            p = paths[condition]
            if os.path.exists(p):
                print(f'  [skip] {condition}')
                continue
            print(f'  generating {condition}...', end=' ', flush=True)
            try:
                if condition == 'baseline':
                    out = image
                elif condition == 'dose_25pct':
                    out = apply_dose_reduction(image, 0.25, rng)
                elif condition == 'dose_50pct':
                    out = apply_dose_reduction(image, 0.50, rng)
                elif condition == 'thick_3mm':
                    out = apply_thick_slices(image, 3.0)
                elif condition == 'thick_5mm':
                    out = apply_thick_slices(image, 5.0)
                elif condition == 'soft_kernel':
                    out = apply_soft_kernel(image)
                sitk.WriteImage(out, p)
                print(f'done ({os.path.getsize(p)/1e6:.1f} MB)')
            except Exception as e:
                print(f'ERROR: {e}')

    print('\nNIfTI generation complete.')


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

PATCH_SIZE = 32   # px
PATCHES_PER_SLICE = 8
LUNG_HU_LO = -950
LUNG_HU_HI = -300


def fft_freq_ratio(patch):
    """High-freq / low-freq power ratio from 2D FFT of patch."""
    f = np.fft.fft2(patch.astype(np.float32))
    f = np.fft.fftshift(f)
    power = np.abs(f) ** 2
    cy, cx = patch.shape[0] // 2, patch.shape[1] // 2
    r = min(cy, cx) // 3
    y, x = np.ogrid[:patch.shape[0], :patch.shape[1]]
    dist = np.sqrt((y - cy)**2 + (x - cx)**2)
    low = power[dist <= r].sum() + 1e-9
    high = power[dist > r].sum() + 1e-9
    return high / low


def corr_ratio_ratio(arr, z, patch_coords):
    """
    Ratio of adjacent-slice correlation to skip-one correlation.
    Thick slices have nearly flat correlation decay (ratio ≈ 1) because
    consecutive averaged slices are nearly identical.
    Thin slices have steeper decay (ratio > 1) since each slice is independent.
    """
    y0, y1, x0, x1 = patch_coords
    if z + 2 >= arr.shape[0]:
        return 1.0  # fallback: neutral ratio
    p0 = arr[z,   y0:y1, x0:x1].astype(np.float32).ravel()
    p1 = arr[z+1, y0:y1, x0:x1].astype(np.float32).ravel()
    p2 = arr[z+2, y0:y1, x0:x1].astype(np.float32).ravel()
    if p0.std() < 1e-6 or p1.std() < 1e-6 or p2.std() < 1e-6:
        return 1.0
    r01 = float(np.corrcoef(p0, p1)[0, 1])
    r02 = float(np.corrcoef(p0, p2)[0, 1])
    # ratio of adjacent to skip-one; clamp to avoid div/zero
    denom = abs(r02) + 1e-6
    return float(r01 / denom)


def extract_features_from_volume(arr):
    """
    arr: 3D numpy array (Z, Y, X) in HU.
    Returns list of (noise_std, edge_sharpness, freq_ratio) tuples,
    one per valid patch.
    """
    features = []
    lung_mask = (arr >= LUNG_HU_LO) & (arr <= LUNG_HU_HI)

    # Sample from middle 60% of slices (avoid apex/base with little lung)
    z_start = int(arr.shape[0] * 0.2)
    z_end   = int(arr.shape[0] * 0.8)
    rng = np.random.default_rng(RANDOM_SEED)

    for z in range(z_start, z_end):
        slc = arr[z]
        mask_slc = lung_mask[z]
        lung_frac = mask_slc.mean()
        if lung_frac < 0.05:  # skip slices with very little lung
            continue

        # Find valid patch positions (lung-rich)
        ys, xs = np.where(mask_slc)
        if len(ys) < PATCH_SIZE * PATCH_SIZE * 0.5:
            continue

        # Sample patch centres from lung pixels
        idxs = rng.choice(len(ys), size=min(PATCHES_PER_SLICE * 4, len(ys)), replace=False)
        count = 0
        for idx in idxs:
            if count >= PATCHES_PER_SLICE:
                break
            cy, cx = int(ys[idx]), int(xs[idx])
            y0, y1 = cy - PATCH_SIZE//2, cy + PATCH_SIZE//2
            x0, x1 = cx - PATCH_SIZE//2, cx + PATCH_SIZE//2
            if y0 < 0 or y1 > slc.shape[0] or x0 < 0 or x1 > slc.shape[1]:
                continue
            patch = slc[y0:y1, x0:x1].astype(np.float32)
            lung_overlap = mask_slc[y0:y1, x0:x1].mean()
            if lung_overlap < 0.5:
                continue

            noise = float(np.std(patch))
            sy = sobel(patch, axis=0)
            sx = sobel(patch, axis=1)
            sharpness = float(np.mean(np.sqrt(sy**2 + sx**2)))
            freq = float(fft_freq_ratio(patch))
            corr = corr_ratio_ratio(arr, z, (y0, y1, x0, x1))
            features.append((noise, sharpness, freq, corr))
            count += 1

    return features


def extract_all_features():
    rows = []

    for case_id in CASES:
        if case_id in LEGACY_NIFTIS:
            cond_paths = dict(LEGACY_NIFTIS[case_id])
            # soft_kernel not in legacy paths — generated to NIFTI_OUT
            cond_paths['soft_kernel'] = os.path.join(NIFTI_OUT, f'{case_id}_soft_kernel.nii.gz')
        else:
            cond_paths = {c: os.path.join(NIFTI_OUT, f'{case_id}_{c}.nii.gz')
                          for c in CONDITIONS}

        for condition, path in cond_paths.items():
            if not os.path.exists(path):
                print(f'  [skip] {case_id} {condition} — file missing')
                continue

            print(f'  extracting {case_id} {condition}...', end=' ', flush=True)
            try:
                img = sitk.ReadImage(path)
                arr = sitk.GetArrayFromImage(img).astype(np.float32)
                feats = extract_features_from_volume(arr)
                for noise, sharpness, freq, corr in feats:
                    rows.append({
                        'case_id':    case_id,
                        'condition':  condition,
                        'noise_std':  noise,
                        'sharpness':  sharpness,
                        'freq_ratio': freq,
                        'corr_ratio': corr,
                    })
                print(f'{len(feats)} patches')
            except Exception as e:
                print(f'ERROR: {e}')

    df = pd.DataFrame(rows)
    df.to_csv(FEATURES_CSV, index=False)
    print(f'\nFeatures saved: {FEATURES_CSV} ({len(df)} patches from {df["case_id"].nunique()} cases)')
    return df


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

THIN_BASELINE_CASES = {
    # cases where baseline z-spacing <= 1.5mm, so thick-slice sim is meaningful
    'LIDC-IDRI-0002',  # 1.25mm
    'LIDC-IDRI-0004',  # 1.25mm
    'LIDC-IDRI-0009',  # ~1.25mm (legacy)
    'LIDC-IDRI-0010',  # 1.25mm
    'LIDC-IDRI-0092',  # 1.25mm
}


def classify(df=None, thin_only=False):
    if df is None:
        print(f'Loading features from {FEATURES_CSV}')
        df = pd.read_csv(FEATURES_CSV)

    if thin_only:
        df = df[df['case_id'].isin(THIN_BASELINE_CASES)].copy()
        print(f'\n--- Thin-baseline cases only ({len(THIN_BASELINE_CASES)} cases) ---')

    print(f'\nDataset: {len(df)} patches, {df["case_id"].nunique()} cases, '
          f'{df["condition"].nunique()} conditions')
    print(f'Condition counts:\n{df["condition"].value_counts().to_string()}')

    X = df[['noise_std', 'sharpness', 'freq_ratio', 'corr_ratio']].values
    y = df['condition'].values
    groups = df['case_id'].values

    # Leave-one-case-out cross-validation
    logo = LeaveOneGroupOut()
    scaler = StandardScaler()
    clf = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED, solver='lbfgs')

    all_true, all_pred = [], []
    case_ids = np.unique(groups)

    for train_idx, test_idx in logo.split(X, y, groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        X_train_s = scaler.fit_transform(X_train)
        X_test_s  = scaler.transform(X_test)
        clf.fit(X_train_s, y_train)
        preds = clf.predict(X_test_s)
        all_true.extend(y_test)
        all_pred.extend(preds)

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)
    accuracy = (all_true == all_pred).mean()

    print(f'\n=== Leave-one-case-out accuracy: {accuracy:.1%} (chance: 20%) ===\n')
    print(classification_report(all_true, all_pred))

    # Per-condition accuracy
    print('Per-condition accuracy:')
    all_conditions = sorted(np.unique(all_true))
    for cond in all_conditions:
        mask = all_true == cond
        if mask.sum() == 0:
            continue
        acc = (all_pred[mask] == cond).mean()
        print(f'  {cond:12s}  {acc:.1%}  (n={mask.sum()})')

    # Feature importance (coefficients from final full-data model)
    X_s = scaler.fit_transform(X)
    clf.fit(X_s, y)
    print('\nFeature coefficients (full model):')
    feat_names = ['noise_std', 'sharpness', 'freq_ratio', 'corr_ratio']
    for i, cond in enumerate(clf.classes_):
        coefs = dict(zip(feat_names, clf.coef_[i]))
        print(f'  {cond:12s}  ' + '  '.join(f'{k}={v:+.3f}' for k, v in coefs.items()))

    return accuracy


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-gen',     action='store_true', help='Skip NIfTI generation')
    parser.add_argument('--skip-extract', action='store_true', help='Skip feature extraction')
    args = parser.parse_args()

    if not args.skip_gen:
        print('=== Step 1: Generate NIfTIs ===')
        generate_niftis()

    if not args.skip_extract:
        print('\n=== Step 2: Extract features ===')
        df = extract_all_features()
    else:
        df = None

    print('\n=== Step 3a: Classify (all 21 cases) ===')
    classify(df, thin_only=False)

    print('\n=== Step 3b: Classify (thin-baseline cases only) ===')
    classify(df, thin_only=True)
