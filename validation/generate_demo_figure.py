"""
GammaMetric — Demo Figure Generator

Generates a publication-quality comparison figure from existing pipeline outputs.
Reads DICOM slices and NIfTI segmentation masks — no reprocessing required.

Usage:
    python generate_demo_figure.py

Point OUTPUT_DIR at any completed gammametric_validation_* folder.
"""

import os
import numpy as np
import pydicom
import SimpleITK as sitk
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path


# ─── CONFIGURATION ────────────────────────────────────────────────────────────

# Point this at your completed validation output folder
OUTPUT_DIR = r"C:\Users\Dan\Desktop\gammametric_output\gammametric_validation_20260306_1638"  # update timestamp

COMPARISON_PROFILE = "thick_slices_5mm"
COMPARISON_LABEL = "5mm Slice Thickness"

ORGAN_COLORS = {
    "lung_upper_lobe_left":   "#00ff88",
    "lung_lower_lobe_left":   "#4488ff",
    "lung_upper_lobe_right":  "#ff8800",
    "lung_lower_lobe_right":  "#ff4488",
    "lung_middle_lobe_right": "#ff3333",
}

# Dice scores from your last run — paste from results.csv
# Format: {organ_display_name: {profile_label: dice}}
DICE_SCORES = {
    "Liver":         {"5mm Slice Thickness": 0.34},
    "Spleen":        {"5mm Slice Thickness": 0.27},
    "Kidney Right":  {"5mm Slice Thickness": None},
    "Kidney Left":   {"5mm Slice Thickness": None},
    "Aorta":         {"5mm Slice Thickness": 0.16},
}


# ─── DICOM LOADING ────────────────────────────────────────────────────────────

def load_dicom_volume(dicom_dir: str):
    """Load DICOM series as a sorted numpy volume."""
    path = Path(dicom_dir)
    files = sorted(path.glob("*.dcm"))
    if not files:
        files = sorted([f for f in path.iterdir() if f.is_file()])

    datasets = []
    for f in files:
        try:
            ds = pydicom.dcmread(str(f))
            datasets.append(ds)
        except Exception:
            pass

    datasets.sort(key=lambda d: float(getattr(d, 'SliceLocation', getattr(d, 'InstanceNumber', 0))))

    slope = float(getattr(datasets[0], 'RescaleSlope', 1))
    intercept = float(getattr(datasets[0], 'RescaleIntercept', 0))

    volume = np.stack([ds.pixel_array.astype(np.float32) * slope + intercept
                       for ds in datasets], axis=0)
    return volume


def load_nifti_mask(seg_dir: str, organ: str):
    """Load organ segmentation mask from NIfTI file."""
    nifti_path = Path(seg_dir) / f"{organ}.nii.gz"
    if not nifti_path.exists():
        nifti_path = Path(seg_dir) / f"{organ}.nii"
    if not nifti_path.exists():
        return None
    img = sitk.ReadImage(str(nifti_path))
    return sitk.GetArrayFromImage(img).astype(np.uint8)


def find_best_slice(masks: dict, volume_shape: tuple) -> int:
    """Find the slice with the most organ content across all masks."""
    combined = np.zeros(volume_shape[0])
    for mask in masks.values():
        if mask is not None:
            n_slices = min(mask.shape[0], volume_shape[0])
            combined[:n_slices] += mask[:n_slices].sum(axis=(1, 2))
    return int(np.argmax(combined))


def window_level(hu_slice: np.ndarray, window: int = 1500, level: int = -600) -> np.ndarray:
    """Apply CT window/level for soft tissue display."""
    low = level - window / 2
    high = level + window / 2
    windowed = np.clip(hu_slice, low, high)
    return ((windowed - low) / (high - low) * 255).astype(np.uint8)


# ─── FIGURE GENERATION ────────────────────────────────────────────────────────

def generate_comparison_figure(output_dir: str, profile_name: str, profile_label: str):
    out = Path(output_dir)

    original_dir = str(out / "original")
    degraded_dir = str(out / f"dicom_{profile_name}")
    baseline_seg_dir = str(out / "seg_baseline")
    degraded_seg_dir = str(out / f"seg_{profile_name}")

    print("Loading original volume...")
    orig_vol = load_dicom_volume(original_dir)

    print("Loading degraded volume...")
    deg_vol = load_dicom_volume(degraded_dir)

    print("Loading segmentation masks...")
    organs = list(ORGAN_COLORS.keys())
    baseline_masks = {o: load_nifti_mask(baseline_seg_dir, o) for o in organs}
    degraded_masks = {o: load_nifti_mask(degraded_seg_dir, o) for o in organs}

    # Find best slice in original volume
    # Find slice with maximum divergence between baseline and degraded masks
    divergence = np.zeros(orig_vol.shape[0])
    for organ in organs:
        bm = baseline_masks.get(organ)
        dm = degraded_masks.get(organ)
        if bm is not None and dm is not None:
            n = min(bm.shape[0], dm.shape[0])
            for i in range(n):
                base_area = bm[i].sum()
                deg_area = dm[i].sum()
                divergence[i] += abs(float(base_area) - float(deg_area))

    best_slice = int(np.argmax(divergence))

    print(f"Using slice: {best_slice}/{orig_vol.shape[0]}")

    scale = deg_vol.shape[0] / orig_vol.shape[0]
    deg_slice_idx = min(int(best_slice * scale), deg_vol.shape[0] - 1)
    orig_slice = window_level(orig_vol[best_slice])
    deg_slice = window_level(deg_vol[deg_slice_idx])


    # ── Build figure ──
    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor('#0e0e0e')

    # Title
    fig.text(0.5, 0.95, 'AI Segmentation Robustness — Imaging Environment Sensitivity',
             ha='center', va='top', color='white', fontsize=15, fontweight='bold')
    fig.text(0.5, 0.91, 'GammaMetric Validation Pipeline  |  TotalSegmentator  |  Chest CT',
             ha='center', va='top', color='#888888', fontsize=9)

    # Two CT panels
    ax1 = fig.add_axes([0.04, 0.12, 0.42, 0.74])
    ax2 = fig.add_axes([0.54, 0.12, 0.42, 0.74])

    for ax in [ax1, ax2]:
        ax.set_facecolor('#0e0e0e')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    # Original CT
    ax1.imshow(orig_slice, cmap='gray', aspect='equal')
    ax1.set_title('Original — 1.25mm Thin Slice', color='white', fontsize=11,
                  pad=8, fontweight='bold')

    # Degraded CT
    ax2.imshow(deg_slice, cmap='gray', aspect='equal')
    ax2.set_title(f'Degraded — {profile_label}', color='#ff6b6b', fontsize=11,
                  pad=8, fontweight='bold')

    # Overlay contours
    for organ, color in ORGAN_COLORS.items():
        # Baseline contours on left panel
        mask = baseline_masks.get(organ)
        if mask is not None and best_slice < mask.shape[0]:
            organ_slice = mask[best_slice]
            if organ_slice.max() > 0:
                ax1.contour(organ_slice, levels=[0.5], colors=[color],
                            linewidths=1.5, alpha=0.9)

        # Degraded contours on right panel
        dmask = degraded_masks.get(organ)
        if dmask is not None and deg_slice_idx < dmask.shape[0]:
            organ_slice = dmask[deg_slice_idx]
            if organ_slice.max() > 0:
                ax2.contour(organ_slice, levels=[0.5], colors=[color],
                            linewidths=1.5, alpha=0.9)

    # Dice score annotations below right panel
    dice_text_lines = []
    organ_display = {
        "lung_upper_lobe_left": "UL Left",
        "lung_lower_lobe_left": "LL Left",
        "lung_upper_lobe_right": "UL Right",
        "lung_lower_lobe_right": "LL Right",
        "lung_middle_lobe_right": "ML Right",
    }


    for organ, label in organ_display.items():
        color = ORGAN_COLORS[organ]
        bm = baseline_masks.get(organ)
        dm = degraded_masks.get(organ)
        if bm is not None and dm is not None:
            n = min(bm.shape[0], dm.shape[0])
            intersection = np.logical_and(bm[:n], dm[:n]).sum()
            total = bm[:n].sum() + dm[:n].sum()
            dice = float(2.0 * intersection / total) if total > 0 else 1.0
            risk_color = '#ff3333' if dice < 0.85 else '#f39c12' if dice < 0.92 else '#27ae60'
            dice_text_lines.append((label, color, f"{dice:.2f}", risk_color))
        else:
            dice_text_lines.append((label, color, "N/A", '#888888'))

    # Dice score bar at bottom
    ax_dice = fig.add_axes([0.54, 0.03, 0.42, 0.07])
    ax_dice.set_facecolor('#1a1a1a')
    ax_dice.set_xticks([])
    ax_dice.set_yticks([])
    for spine in ax_dice.spines.values():
        spine.set_edgecolor('#333333')

    n = len(dice_text_lines)
    for i, (organ_label, organ_color, dice_val, risk_color) in enumerate(dice_text_lines):
        x = (i + 0.5) / n
        ax_dice.text(x, 0.75, organ_label, ha='center', va='center',
                     color=organ_color, fontsize=8, fontweight='bold',
                     transform=ax_dice.transAxes)
        ax_dice.text(x, 0.25, f"Dice: {dice_val}", ha='center', va='center',
                     color=risk_color, fontsize=9, fontweight='bold',
                     transform=ax_dice.transAxes)

    # Legend bottom left
    ax_legend = fig.add_axes([0.04, 0.03, 0.42, 0.07])
    ax_legend.set_facecolor('#1a1a1a')
    ax_legend.set_xticks([])
    ax_legend.set_yticks([])
    for spine in ax_legend.spines.values():
        spine.set_edgecolor('#333333')

    ax_legend.text(0.02, 0.5, 'Segmentation contours:', va='center',
                   color='#888888', fontsize=8, transform=ax_legend.transAxes)

    patches = [mpatches.Patch(color=c, label=organ_display[o])
               for o, c in ORGAN_COLORS.items()]
    ax_legend.legend(handles=patches, loc='center right', ncol=5,
                     frameon=False, fontsize=8,
                     labelcolor='white', bbox_to_anchor=(1.0, 0.5))

    # GammaMetric watermark
    fig.text(0.99, 0.01, 'GammaMetric  |  gammametric.com',
             ha='right', va='bottom', color='#444444', fontsize=7)

    # Save
    out_path = str(out / f"demo_comparison_{profile_name}.png")
    plt.savefig(out_path, dpi=180, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"\nDemo figure saved → {out_path}")
    return out_path


# ─── ENTRY POINT ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    OUTPUT_DIR = input("Enter path to validation output folder: ").strip().strip('"')
    generate_comparison_figure(OUTPUT_DIR, COMPARISON_PROFILE, COMPARISON_LABEL)
python -c "from monai.apps.detection.networks.retinanet_detector import RetinaNetDetector; print('MONAI detection ready')"