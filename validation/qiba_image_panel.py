"""QIBA clinical image comparison panel.

Shows FBP vs ASIR 30/50/70% on the same phantom slice.
ConvolutionKernel = 'STANDARD' for all panels — the tag is identical.
Also shows a GE vs Siemens cross-vendor panel (same recon class).

Outputs:
  panel_GE_ASIR.png       4-panel: FBP / ASIR 30 / ASIR 50 / ASIR 70
  panel_difference.png    difference images vs FBP (what the AI sees differently)
  panel_crossvendor.png   GE STANDARD FBP vs Siemens B30f (closest equivalent)

Usage:
  python validation/qiba_image_panel.py \
      --inv  G:/GammaMetric/qiba_inventory_out/series_inventory.csv \
      --root G:/GammaMetric/manifest-1619103006849/QIBA-CT-Liver-Phantom \
      --out  G:/GammaMetric/qiba_nps_out
"""
from __future__ import annotations

import os
import argparse

import numpy as np
import pandas as pd
import pydicom
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Window / level for display (abdomen soft tissue)
WL, WW = 50, 350   # centre, width  ->  display range = WL ± WW/2

# Series to use (2.5mm GE, replicate 1 where available)
GE_SERIES = {
    "FBP":     "1.2.840.113619.2.340.3.1930077730.508.1439460703.218.12",
    "ASIR 30%": "1.2.840.113619.2.340.3.1930077730.508.1439460703.218.15",
    "ASIR 50%": "1.2.840.113619.2.340.3.1930077730.508.1439460703.218.18",
    "ASIR 70%": "1.2.840.113619.2.340.3.1930077730.199.1439484877.320",
}

# Siemens B30f, 250mAs — pick first available 1.5mm series (closest to GE 2.5mm)
SIE_KERNEL_TARGET = "B30f"
SIE_MAS_TARGET    = "250"
SIE_SLICE_TARGET  = "1.5"


def build_uid_map(root: str) -> dict[str, str]:
    uid_to_dir = {}
    for dirpath, _dirs, files in os.walk(root):
        for fn in files:
            p = os.path.join(dirpath, fn)
            try:
                ds = pydicom.dcmread(p, stop_before_pixels=True,
                                     specific_tags=["SeriesInstanceUID"])
                uid = str(ds.get("SeriesInstanceUID", ""))
                if uid and uid not in uid_to_dir:
                    uid_to_dir[uid] = dirpath
            except Exception:
                continue
    return uid_to_dir


def load_mid_slice(dicom_dir: str) -> np.ndarray:
    """Load the central HU slice from a series directory."""
    entries = []
    for fn in os.listdir(dicom_dir):
        p = os.path.join(dicom_dir, fn)
        try:
            ds = pydicom.dcmread(p, stop_before_pixels=False)
            if not hasattr(ds, "pixel_array"):
                continue
            pos = float(ds.ImagePositionPatient[2]) \
                  if hasattr(ds, "ImagePositionPatient") else float(getattr(ds, "InstanceNumber", 0))
            entries.append((pos, ds))
        except Exception:
            continue
    if not entries:
        raise ValueError(f"No readable DICOM in {dicom_dir}")
    entries.sort(key=lambda x: x[0])
    _, ds = entries[len(entries) // 2]
    arr = ds.pixel_array.astype(np.float32)
    slope     = float(getattr(ds, "RescaleSlope", 1))
    intercept = float(getattr(ds, "RescaleIntercept", 0))
    return arr * slope + intercept


def window(img: np.ndarray, wl: float, ww: float) -> np.ndarray:
    lo, hi = wl - ww / 2, wl + ww / 2
    return np.clip((img - lo) / (hi - lo), 0, 1)


def crop_centre(img: np.ndarray, size: int = 256) -> np.ndarray:
    h, w = img.shape
    r0 = max(0, h // 2 - size // 2)
    c0 = max(0, w // 2 - size // 2)
    return img[r0:r0 + size, c0:c0 + size]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inv",  required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out",  default="qiba_nps_out")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    inv = pd.read_csv(args.inv, dtype=str).fillna("")
    print("Building UID->dir map...")
    uid_map = build_uid_map(args.root)

    # ---- Load GE ASIR series ------------------------------------------------
    slices = {}
    for label, uid in GE_SERIES.items():
        d = uid_map.get(uid)
        if not d:
            print(f"  ! UID not found for {label}")
            continue
        s = load_mid_slice(d)
        slices[label] = crop_centre(s, 320)
        print(f"  loaded {label}: {slices[label].shape}, HU range {slices[label].min():.0f}–{slices[label].max():.0f}")

    # ---- Panel 1: FBP / ASIR 30 / 50 / 70 ----------------------------------
    order = ["FBP", "ASIR 30%", "ASIR 50%", "ASIR 70%"]
    available = [k for k in order if k in slices]
    if available:
        fig, axes = plt.subplots(1, len(available), figsize=(4 * len(available), 4.5))
        if len(available) == 1:
            axes = [axes]
        for ax, label in zip(axes, available):
            ax.imshow(window(slices[label], WL, WW), cmap="gray", vmin=0, vmax=1,
                      interpolation="nearest")
            ax.set_title(label, fontsize=11, fontweight="bold")
            ax.axis("off")
            # tag annotation
            ax.text(0.02, 0.03, "ConvolutionKernel = 'STANDARD'",
                    transform=ax.transAxes, fontsize=6.5, color="yellow",
                    va="bottom", ha="left",
                    bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.6))
        fig.suptitle(
            "GE Discovery CT750 HD — 275 mA, 2.5 mm, STANDARD kernel\n"
            "Same ConvolutionKernel tag (0018,1210) for all panels",
            fontsize=10, y=1.01)
        fig.tight_layout()
        out1 = os.path.join(args.out, "panel_GE_ASIR.png")
        fig.savefig(out1, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out1}")

    # ---- Panel 2: difference images (ASIR N - FBP) --------------------------
    if "FBP" in slices:
        diff_labels = [k for k in ["ASIR 30%", "ASIR 50%", "ASIR 70%"] if k in slices]
        if diff_labels:
            fig, axes = plt.subplots(1, len(diff_labels), figsize=(4 * len(diff_labels), 4.5))
            if len(diff_labels) == 1:
                axes = [axes]
            # symmetric colour scale based on max abs diff
            diffs = [slices[k] - slices["FBP"] for k in diff_labels]
            vmax = max(np.abs(d).max() for d in diffs)
            vmax = min(vmax, 200)   # cap for display
            for ax, label, diff in zip(axes, diff_labels, diffs):
                im = ax.imshow(diff, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                               interpolation="nearest")
                ax.set_title(f"{label} − FBP", fontsize=11, fontweight="bold")
                ax.axis("off")
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="ΔHU")
            fig.suptitle(
                "Pixel difference from FBP — what iterative recon changes\n"
                "(same ConvolutionKernel tag; AI trained on FBP sees these as distribution shift)",
                fontsize=9, y=1.02)
            fig.tight_layout()
            out2 = os.path.join(args.out, "panel_difference.png")
            fig.savefig(out2, dpi=180, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved {out2}")

    # ---- Panel 3: cross-vendor GE FBP vs Siemens B30f -----------------------
    sie_rows = inv[
        inv["Manufacturer"].str.upper().str.contains("SIEMENS") &
        (inv["ConvolutionKernel"] == SIE_KERNEL_TARGET) &
        (inv["Exposure"] == SIE_MAS_TARGET) &
        (inv["SliceThickness"] == SIE_SLICE_TARGET)
    ]
    if not sie_rows.empty and "FBP" in slices:
        uid_sie = sie_rows.iloc[0]["SeriesInstanceUID"]
        d_sie = uid_map.get(uid_sie)
        if d_sie:
            sie_slice = crop_centre(load_mid_slice(d_sie), 320)
            print(f"  loaded Siemens B30f: {sie_slice.shape}")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.5))
            ax1.imshow(window(slices["FBP"], WL, WW), cmap="gray", vmin=0, vmax=1,
                       interpolation="nearest")
            ax1.set_title("GE — STANDARD FBP", fontsize=11, fontweight="bold")
            ax1.axis("off")
            ax1.text(0.02, 0.03, "ConvKernel = 'STANDARD'",
                     transform=ax1.transAxes, fontsize=7, color="yellow",
                     va="bottom", bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.6))
            ax2.imshow(window(sie_slice, WL, WW), cmap="gray", vmin=0, vmax=1,
                       interpolation="nearest")
            ax2.set_title("Siemens — B30f FBP", fontsize=11, fontweight="bold")
            ax2.axis("off")
            ax2.text(0.02, 0.03, "ConvKernel = 'B30f'",
                     transform=ax2.transAxes, fontsize=7, color="yellow",
                     va="bottom", bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.6))
            fig.suptitle(
                "Cross-vendor comparison — same phantom, similar protocol\n"
                "Kernel tags are vendor-proprietary strings with no common reference standard",
                fontsize=9, y=1.01)
            fig.tight_layout()
            out3 = os.path.join(args.out, "panel_crossvendor.png")
            fig.savefig(out3, dpi=180, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved {out3}")
        else:
            print("  ! Siemens B30f dir not found in UID map")
    else:
        print("  Skipping cross-vendor panel (no Siemens B30f 1.5mm series or FBP not loaded)")

    print("\nDone.")


if __name__ == "__main__":
    main()
