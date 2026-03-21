"""
GammaMetric — AI Robustness Validation Pipeline v0.2

Pipeline:
    1. Load original DICOM CT series
    2. Run TotalSegmentator → baseline segmentation
    3. Apply degradation profiles (dose, kernel, slice thickness)
    4. Run TotalSegmentator → degraded segmentations
    5. Compute Dice, volume difference, boundary shift per organ
    6. Generate summary report with risk scoring

Dependencies:
    pip install pydicom numpy scipy SimpleITK totalsegmentator matplotlib pandas

Usage:
    python gammametric_validation_pipeline.py <input_dicom_dir> <output_dir>
"""

import os
import copy
import json
import subprocess
import numpy as np
import pydicom
import SimpleITK as sitk
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import gaussian_filter, zoom
from pathlib import Path
from datetime import datetime


# ─── CONFIGURATION ────────────────────────────────────────────────────────────

# Organs TotalSegmentator reports — subset most relevant to robustness demo
TARGET_ORGANS = [
    "lung_upper_lobe_left",
    "lung_lower_lobe_left",
    "lung_upper_lobe_right",
    "lung_lower_lobe_right",
    "lung_middle_lobe_right"
]



# Degradation profiles — three clinical dimensions
DEGRADATION_PROFILES = {
    # A. Dose reduction / noise injection
    "dose_reduction_50pct": {
        "type": "dose",
        "dose_factor": 0.50,
        "noise_sigma": 20,
        "blur_sigma": 0.0,
        "slice_resample": None,
        "label": "50% Dose Reduction",
        "clinical_context": "Aggressive dose optimization or underexposed scan"
    },
    "dose_reduction_25pct": {
        "type": "dose",
        "dose_factor": 0.25,
        "noise_sigma": 40,
        "blur_sigma": 0.0,
        "slice_resample": None,
        "label": "75% Dose Reduction",
        "clinical_context": "Ultra-low dose protocol or dose drift scenario"
    },
    # B. Reconstruction kernel simulation
    "soft_kernel": {
        "type": "kernel",
        "dose_factor": 1.0,
        "noise_sigma": 0,
        "blur_sigma": 1.5,
        "slice_resample": None,
        "label": "Soft Reconstruction Kernel",
        "clinical_context": "Vendor kernel variation or soft tissue reconstruction"
    },
    "sharp_kernel": {
        "type": "kernel",
        "dose_factor": 1.0,
        "noise_sigma": 5,
        "blur_sigma": 0.3,
        "slice_resample": None,
        "label": "Sharp Reconstruction Kernel",
        "clinical_context": "High-frequency kernel, edge-enhancing reconstruction"
    },
    # C. Slice thickness resampling
    "thick_slices_3mm": {
        "type": "resolution",
        "dose_factor": 1.0,
        "noise_sigma": 0,
        "blur_sigma": 0.0,
        "slice_resample": 3.0,
        "label": "3mm Slice Thickness",
        "clinical_context": "Older scanner or non-standard acquisition protocol"
    },
    "thick_slices_5mm": {
        "type": "resolution",
        "dose_factor": 1.0,
        "noise_sigma": 0,
        "blur_sigma": 0.0,
        "slice_resample": 5.0,
        "label": "5mm Slice Thickness",
        "clinical_context": "Legacy protocol or radiation dose-sparing acquisition"
    },
}

# Risk thresholds for Dice score
RISK_THRESHOLDS = {
    "HIGH":     0.85,   # Dice < 0.85 = high sensitivity
    "MODERATE": 0.92,   # Dice 0.85-0.92 = moderate
    "LOW":      1.00,   # Dice > 0.92 = low sensitivity
}


# ─── DEGRADATION ENGINE ───────────────────────────────────────────────────────

def add_poisson_noise(pixel_array: np.ndarray, dose_factor: float) -> np.ndarray:
    """Simulate dose reduction via Poisson noise scaling."""
    shift = abs(pixel_array.min()) + 1
    shifted = pixel_array.astype(np.float32) + shift
    scaled = shifted * dose_factor
    noisy = np.random.poisson(np.maximum(scaled, 0)).astype(np.float32)
    result = (noisy / dose_factor) - shift
    return np.clip(result, -32768, 32767).astype(np.int16)


def add_gaussian_noise(pixel_array: np.ndarray, sigma: float) -> np.ndarray:
    """Add Gaussian noise to simulate increased image noise."""
    noise = np.random.normal(0, sigma, pixel_array.shape)
    noisy = pixel_array.astype(np.float32) + noise
    return np.clip(noisy, -32768, 32767).astype(np.int16)


def apply_blur(pixel_array: np.ndarray, sigma: float) -> np.ndarray:
    """Apply Gaussian blur to simulate soft/sharp kernel variation."""
    blurred = gaussian_filter(pixel_array.astype(np.float32), sigma=sigma)
    return np.clip(blurred, -32768, 32767).astype(np.int16)


def resample_slice_thickness(
    datasets: list,
    target_thickness_mm: float
) -> list:
    """
    Resample DICOM series to simulate thicker slice acquisition.
    Uses averaging of adjacent slices to simulate partial volume effect.
    """
    if not datasets:
        return datasets

    # Get original slice thickness
    orig_thickness = float(getattr(datasets[0], 'SliceThickness', 1.0))
    if orig_thickness <= 0:
        orig_thickness = 1.0

    resample_factor = orig_thickness / target_thickness_mm
    if resample_factor >= 1.0:
        print(f"  Target thickness {target_thickness_mm}mm <= original {orig_thickness}mm, skipping resample")
        return datasets

    n_orig = len(datasets)
    n_new = max(1, int(n_orig * resample_factor))

    # Build 3D volume
    volume = np.stack([ds.pixel_array for ds in datasets], axis=0).astype(np.float32)

    # Zoom along slice axis
    zoom_factors = (resample_factor, 1.0, 1.0)
    resampled_volume = zoom(volume, zoom_factors, order=1)

    # Build new dataset list from resampled volume
    new_datasets = []
    step = max(1, n_orig // n_new)
    for i in range(min(n_new, resampled_volume.shape[0])):
        ds = copy.deepcopy(datasets[min(i * step, n_orig - 1)])
        slice_data = np.clip(resampled_volume[i], -32768, 32767).astype(np.int16)
        ds.PixelData = slice_data.tobytes()
        ds.SliceThickness = target_thickness_mm
        ds.InstanceNumber = i + 1
        new_datasets.append(ds)

    print(f"  Resampled {n_orig} slices → {len(new_datasets)} slices ({target_thickness_mm}mm)")
    return new_datasets


def apply_degradation_profile(
    datasets: list,
    profile: dict
) -> list:
    """Apply a degradation profile to a DICOM series."""
    result = copy.deepcopy(datasets)

    # Slice thickness resampling first (changes number of slices)
    if profile.get("slice_resample"):
        result = resample_slice_thickness(result, profile["slice_resample"])

    # Per-slice degradations
    for ds in result:
        pixel_array = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, 'RescaleSlope', 1))
        intercept = float(getattr(ds, 'RescaleIntercept', 0))
        hu = (pixel_array * slope + intercept).astype(np.int16)

        if profile["dose_factor"] < 1.0:
            hu = add_poisson_noise(hu, profile["dose_factor"])
        if profile["noise_sigma"] > 0:
            hu = add_gaussian_noise(hu, profile["noise_sigma"])
        if profile["blur_sigma"] > 0:
            hu = apply_blur(hu, profile["blur_sigma"])

        raw = np.clip(((hu.astype(np.float32) - intercept) / slope), -32768, 32767).astype(np.int16)
        ds.PixelData = raw.tobytes()

    return result


# ─── DICOM I/O ────────────────────────────────────────────────────────────────

def load_dicom_series(series_dir: str) -> list:
    """Load and sort DICOM series from directory."""
    path = Path(series_dir)
    files = list(path.glob("*.dcm"))
    if not files:
        files = [f for f in path.iterdir() if f.is_file() and not f.name.startswith('.')]

    datasets = []
    for f in sorted(files):
        try:
            ds = pydicom.dcmread(str(f))
            datasets.append(ds)
        except Exception as e:
            print(f"  Warning: skipping {f.name}: {e}")

    datasets.sort(key=lambda d: int(getattr(d, 'InstanceNumber', 0)))
    print(f"Loaded {len(datasets)} slices from {series_dir}")
    return datasets


def save_dicom_series(datasets: list, output_dir: str, tag: str = "degraded"):
    """Save DICOM series to directory."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for i, ds in enumerate(datasets):
        ds.SeriesDescription = f"{getattr(ds, 'SeriesDescription', 'Series')} [GM:{tag}]"
        out_path = os.path.join(output_dir, f"{tag}_{i+1:04d}.dcm")
        ds.save_as(out_path)
    return output_dir


# ─── TOTALSEGMENTATOR INTEGRATION ────────────────────────────────────────────

def run_totalsegmentator(dicom_dir: str, output_dir: str) -> bool:
    """
    Run TotalSegmentator on a DICOM directory.
    Outputs NIfTI segmentation masks to output_dir.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    cmd = [
        "TotalSegmentator",
        "-i", dicom_dir,
        "-o", output_dir,
        "--fast",           # Fast mode for demo — use full for production

    ]
    print(f"  Running TotalSegmentator on {dicom_dir}...")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            print(f"  TotalSegmentator error: {result.stderr}")
            return False
        print(f"  Segmentation complete → {output_dir}")
        return True
    except subprocess.TimeoutExpired:
        print("  TotalSegmentator timed out")
        return False
    except FileNotFoundError:
        print("  TotalSegmentator not found — install with: pip install totalsegmentator")
        return False


# ─── COMPARISON ENGINE ────────────────────────────────────────────────────────

def load_segmentation_mask(seg_dir: str, organ: str) -> np.ndarray | None:
    """Load a NIfTI segmentation mask for a specific organ."""
    nifti_path = Path(seg_dir) / f"{organ}.nii.gz"
    if not nifti_path.exists():
        nifti_path = Path(seg_dir) / f"{organ}.nii"
    if not nifti_path.exists():
        print(f"  Mask not found: {nifti_path}")
        return None

    img = sitk.ReadImage(str(nifti_path))
    return sitk.GetArrayFromImage(img).astype(np.uint8)




def compute_dice(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Compute Dice similarity coefficient between two binary masks."""
    if mask_a is None or mask_b is None:
        return None

    # Align shapes if resampling changed slice count
    min_slices = min(mask_a.shape[0], mask_b.shape[0])
    mask_a = mask_a[:min_slices]
    mask_b = mask_b[:min_slices]

    intersection = np.logical_and(mask_a, mask_b).sum()
    total = mask_a.sum() + mask_b.sum()
    if total == 0:
        return 1.0
    return float(2.0 * intersection / total)


def compute_volume_difference_pct(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    if mask_a is None or mask_b is None:
        return None
    vol_a = float(mask_a.sum())
    if vol_a == 0:
        return None
    return float((float(mask_b.sum()) - vol_a) / vol_a * 100)


def risk_label(dice: float) -> str:
    """Assign risk label based on Dice score."""
    if dice is None:
        return "N/A"
    if dice < RISK_THRESHOLDS["HIGH"]:
        return "HIGH"
    elif dice < RISK_THRESHOLDS["MODERATE"]:
        return "MODERATE"
    else:
        return "LOW"


def compare_segmentations(
    baseline_seg_dir: str,
    degraded_seg_dir: str,
    profile_label: str
) -> list[dict]:
    """Compare TotalSegmentator outputs between baseline and degraded series."""
    results = []
    for organ in TARGET_ORGANS:
        baseline_mask = load_segmentation_mask(baseline_seg_dir, organ)
        degraded_mask = load_segmentation_mask(degraded_seg_dir, organ)

        dice = compute_dice(baseline_mask, degraded_mask)
        vol_diff = compute_volume_difference_pct(baseline_mask, degraded_mask)

        results.append({
            "organ": organ.replace("_", " ").title(),
            "profile": profile_label,
            "dice": round(dice, 3) if dice is not None else None,
            "volume_diff_pct": round(vol_diff, 1) if vol_diff is not None else None,
            "risk": risk_label(dice)
        })
        dice_str = f"{dice:.3f}" if dice is not None else "N/A"
        vol_str = f"{vol_diff:.1f}" if vol_diff is not None else "N/A"
        print(f"    {organ}: Dice={dice_str}, VolDiff={vol_str}%, Risk={risk_label(dice)}")
    return results


# ─── VISUALIZATION ────────────────────────────────────────────────────────────

def generate_dice_heatmap(all_results: list[dict], output_path: str):
    """Generate a Dice score heatmap across organs and degradation profiles."""
    df = pd.DataFrame(all_results)
    pivot = df.pivot(index="organ", columns="profile", values="dice")

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor('#1a1a1a')
    ax.set_facecolor('#1a1a1a')

    cmap = plt.cm.RdYlGn
    im = ax.imshow(pivot.values.astype(float), cmap=cmap, vmin=0.7, vmax=1.0, aspect='auto')

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha='right', color='white', fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, color='white', fontsize=10)

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if val is not None and not np.isnan(float(val)):
                color = 'black' if float(val) > 0.9 else 'white'
                ax.text(j, i, f"{float(val):.2f}", ha='center', va='center',
                        color=color, fontsize=9, fontweight='bold')

    plt.colorbar(im, ax=ax, label='Dice Score').ax.yaxis.label.set_color('white')
    ax.set_title('AI Segmentation Robustness — Dice Score by Organ & Degradation Profile\n'
                 'GammaMetric Validation Pipeline', color='white', fontsize=12, pad=15)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"Heatmap saved → {output_path}")


def generate_risk_summary(all_results: list[dict], output_path: str):
    """Generate a traffic light risk summary suitable for non-technical audience."""
    df = pd.DataFrame(all_results)

    # Aggregate risk per profile
    profile_risk = {}
    for profile in df["profile"].unique():
        subset = df[df["profile"] == profile]
        high_count = (subset["risk"] == "HIGH").sum()
        mod_count = (subset["risk"] == "MODERATE").sum()
        if high_count > 0:
            overall = "HIGH"
        elif mod_count > 1:
            overall = "MODERATE"
        else:
            overall = "LOW"
        profile_risk[profile] = overall

    fig, ax = plt.subplots(figsize=(10, max(4, len(profile_risk) * 0.8)))
    fig.patch.set_facecolor('#1a1a1a')
    ax.set_facecolor('#1a1a1a')
    ax.axis('off')

    colors = {"HIGH": "#c0392b", "MODERATE": "#f39c12", "LOW": "#27ae60"}
    headers = ["Degradation Scenario", "Clinical Context", "AI Sensitivity"]
    col_widths = [0.35, 0.45, 0.2]

    # Header
    x = 0
    for h, w in zip(headers, col_widths):
        ax.text(x + w/2, 1.05, h, transform=ax.transAxes,
                ha='center', va='bottom', color='white',
                fontsize=10, fontweight='bold')
        x += w

    for i, (profile_name, risk) in enumerate(profile_risk.items()):
        y = 0.9 - i * (0.85 / max(len(profile_risk), 1))
        profile_data = DEGRADATION_PROFILES.get(profile_name, {})
        context = profile_data.get("clinical_context", "")
        color = colors[risk]

        ax.text(0.01, y, profile_data.get("label", profile_name),
                transform=ax.transAxes, ha='left', va='center',
                color='white', fontsize=9)
        ax.text(0.36, y, context,
                transform=ax.transAxes, ha='left', va='center',
                color='#aaaaaa', fontsize=8)

        # Traffic light circle
        circle = plt.Circle((0.88, y), 0.03, color=color,
                             transform=ax.transAxes, clip_on=False)
        ax.add_patch(circle)
        ax.text(0.88, y, risk, transform=ax.transAxes,
                ha='center', va='center', color='white',
                fontsize=7, fontweight='bold')

    ax.set_title('AI Robustness Risk Summary — GammaMetric\n'
                 'Sensitivity of segmentation model to imaging environment variation',
                 color='white', fontsize=11, pad=15)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"Risk summary saved → {output_path}")


# ─── MAIN PIPELINE ────────────────────────────────────────────────────────────

def run_pipeline(input_dicom_dir: str, output_base_dir: str):
    """
    Full end-to-end validation pipeline.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    out = Path(output_base_dir) / f"gammametric_validation_{timestamp}"
    out.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("GammaMetric — AI Robustness Validation Pipeline")
    print("="*60)

    # Step 1 — Load original series
    print("\n[1/5] Loading original DICOM series...")
    datasets = load_dicom_series(input_dicom_dir)
    if not datasets:
        print("No DICOM files found. Exiting.")
        return

    # Step 2 — Baseline segmentation
    print("\n[2/5] Running baseline segmentation...")
    baseline_dicom_dir = str(out / "original")
    save_dicom_series(datasets, baseline_dicom_dir, tag="original")
    baseline_seg_dir = str(out / "seg_baseline")
    baseline_ok = run_totalsegmentator(baseline_dicom_dir, baseline_seg_dir)

    if not baseline_ok:
        print("\nBaseline segmentation failed.")
        print("To proceed without TotalSegmentator installed,")
        print("run degradation only and inspect outputs in Kevin's viewer.")
        degradation_only(datasets, out)
        return

    # Step 3-4 — Degrade and segment
    print("\n[3/5] Applying degradation profiles and segmenting...")
    all_results = []

    for profile_name, profile in DEGRADATION_PROFILES.items():
        print(f"\n  Profile: {profile['label']}")
        degraded = apply_degradation_profile(datasets, profile)
        degraded_dicom_dir = str(out / f"dicom_{profile_name}")
        save_dicom_series(degraded, degraded_dicom_dir, tag=profile_name)

        degraded_seg_dir = str(out / f"seg_{profile_name}")
        seg_ok = run_totalsegmentator(degraded_dicom_dir, degraded_seg_dir)

        if seg_ok:
            results = compare_segmentations(
                baseline_seg_dir,
                degraded_seg_dir,
                profile["label"]
            )
            all_results.extend(results)

    # Step 5 — Report
    print("\n[4/5] Generating visualizations...")
    generate_dice_heatmap(all_results, str(out / "dice_heatmap.png"))
    generate_risk_summary(all_results, str(out / "risk_summary.png"))

    # Save raw results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(str(out / "results.csv"), index=False)

    print("\n[5/5] Pipeline complete.")
    print(f"\nOutputs saved to: {out}")
    print("  dice_heatmap.png  — Dice scores across organs and profiles")
    print("  risk_summary.png  — Traffic light summary for non-technical audience")
    print("  results.csv       — Raw data")
    print("\nLoad DICOM folders in a viewer to visually compare segmentation overlays.")


def degradation_only(datasets: list, out: Path):
    """Fallback — generate degraded DICOM series without segmentation."""
    print("\nRunning degradation-only mode (no TotalSegmentator)...")
    for profile_name, profile in DEGRADATION_PROFILES.items():
        print(f"  Applying: {profile['label']}")
        degraded = apply_degradation_profile(datasets, profile)
        save_dicom_series(degraded, str(out / f"dicom_{profile_name}"), tag=profile_name)
    print(f"\nDegraded series saved to {out}")
    print("Load in your DICOM viewer to inspect visually.")


#
# )─── ENTRY POINT ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    path = r"C:\Users\Dan\Desktop\gammametric_output\Lung DICOM\manifest-1600709154662\LIDC-IDRI\LIDC-IDRI-0005\01-01-2000-NA-NA-42125\3000548.000000-NA-86225"
    run_pipeline(path, r"C:\Users\Dan\Desktop\gammametric_output")