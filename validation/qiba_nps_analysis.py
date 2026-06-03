"""QIBA-CT-Liver-Phantom — NPS and feature-space analysis.

Demonstrates that iterative reconstruction strength (GE ASIR, Siemens SAFIRE)
creates measurable pixel-level differences that the standard ConvolutionKernel
DICOM tag cannot capture.

Key experiment:
  GE:      STANDARD kernel, 275 mA, all ASIR levels (FBP / 30 / 50 / 70%)
           → ConvolutionKernel = "STANDARD" throughout; only private tag differs
  Siemens: B20f vs B20s, 250 mAs
           → FBP vs SAFIRE; kernel tag differs by one undocumented character

Outputs (--out dir):
  nps_curves_GE.png         NPS H(f) by ASIR level, GE STANDARD kernel
  nps_curves_Siemens.png    NPS H(f) B20f vs B20s
  feature_pca_GE.png        PCA of fingerprint features, colored by ASIR
  feature_pca_Siemens.png   PCA, colored by kernel
  nps_table.csv             NPS peak frequency + peak value per series
  features.csv              per-series fingerprint feature vectors

Usage:
  python validation/qiba_nps_analysis.py \
      --inv  G:/GammaMetric/qiba_inventory_out/series_inventory.csv \
      --root G:/GammaMetric/manifest-1619103006849/QIBA-CT-Liver-Phantom \
      --out  G:/GammaMetric/qiba_nps_out
"""
from __future__ import annotations

import os
import csv
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
import pydicom
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import sobel
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------- #
# Series selection criteria
# --------------------------------------------------------------------------- #
GE_TARGET_KERNEL  = "STANDARD"
GE_TARGET_MA      = "275"          # matches desc_mA column
GE_ASIR_LEVELS    = ["", "30", "50", "70"]   # "" = FBP (no ASIR in desc)
SIE_TARGET_MAS    = "250"          # Exposure mAs column
SIE_KERNELS       = ["B20f", "B20s", "B30f", "B30s"]

# NPS computation parameters
ROI_SIZE     = 64    # pixels, square ROI for NPS
N_ROIS       = 16    # ROIs sampled per slice
N_SLICES     = 10    # central slices used per series
HU_MIN       = -100  # phantom background (uniform surround, not liver inserts)
HU_MAX       = 100


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def load_series_sorted(dicom_dir: str) -> list[np.ndarray]:
    """Load all slices from a directory, sorted by InstanceNumber / ImagePositionPatient."""
    files = []
    for fn in os.listdir(dicom_dir):
        p = os.path.join(dicom_dir, fn)
        try:
            ds = pydicom.dcmread(p, stop_before_pixels=False)
            if not hasattr(ds, "pixel_array"):
                continue
            pos = float(ds.get("ImagePositionPatient", [0, 0, 0])[2]) \
                  if ds.get("ImagePositionPatient") else float(ds.get("InstanceNumber", 0))
            files.append((pos, ds))
        except Exception:
            continue
    files.sort(key=lambda x: x[0])
    slices = []
    for _, ds in files:
        arr = ds.pixel_array.astype(np.float32)
        slope  = float(getattr(ds, "RescaleSlope",  1))
        intercept = float(getattr(ds, "RescaleIntercept", 0))
        slices.append(arr * slope + intercept)
    return slices


def sample_rois(slices: list[np.ndarray], n_slices: int, roi_size: int,
                n_rois: int, hu_min: float, hu_max: float) -> np.ndarray:
    """Return (N, roi_size, roi_size) array of background ROIs."""
    mid = len(slices) // 2
    selected = slices[max(0, mid - n_slices // 2): mid + n_slices // 2]
    rois = []
    rng = np.random.default_rng(42)
    for sl in selected:
        H, W = sl.shape
        attempts = 0
        found = 0
        while found < n_rois and attempts < n_rois * 20:
            r = rng.integers(0, H - roi_size)
            c = rng.integers(0, W - roi_size)
            patch = sl[r:r + roi_size, c:c + roi_size]
            if hu_min < patch.mean() < hu_max and patch.std() < 40:
                rois.append(patch)
                found += 1
            attempts += 1
    return np.array(rois) if rois else np.zeros((1, roi_size, roi_size))


def compute_nps_2d(rois: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Radially-averaged 2D NPS from ROI ensemble. Returns (freq, nps)."""
    n, H, W = rois.shape
    psds = []
    for roi in rois:
        detrended = roi - roi.mean()
        ft = np.fft.fft2(detrended)
        psd = (np.abs(ft) ** 2) / (H * W)
        psds.append(np.fft.fftshift(psd))
    mean_psd = np.mean(psds, axis=0)

    # radial average
    cy, cx = H // 2, W // 2
    y, x = np.ogrid[:H, :W]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(int)
    max_r = min(cx, cy)
    nps_radial = np.array([mean_psd[r == i].mean() if (r == i).any() else 0
                           for i in range(max_r)])
    # assume pixel spacing = 1 (relative); actual spacing applied later if available
    freq = np.arange(max_r) / (H)
    return freq, nps_radial


def fingerprint_features(rois: np.ndarray) -> dict:
    """Compute the same 4 features as acquisition_fingerprint.py."""
    noise_stds, sharpnesses, freq_ratios, corr_ratios = [], [], [], []
    for roi in rois:
        noise_stds.append(roi.std())
        gx = sobel(roi, axis=0)
        gy = sobel(roi, axis=1)
        sharpnesses.append(np.sqrt(gx**2 + gy**2).mean())
        ft = np.abs(np.fft.fft2(roi))
        half = roi.shape[0] // 2
        low  = ft[:half // 2, :half // 2].mean() + 1e-9
        high = ft[half // 2:, half // 2:].mean() + 1e-9
        freq_ratios.append(high / low)
        flat = roi.flatten()
        if len(flat) > 1:
            corr = np.corrcoef(flat[:-1], flat[1:])[0, 1]
        else:
            corr = 0.0
        corr_ratios.append(corr)
    return {
        "noise_std":  np.mean(noise_stds),
        "sharpness":  np.mean(sharpnesses),
        "freq_ratio": np.mean(freq_ratios),
        "corr_ratio": np.mean(corr_ratios),
    }


# --------------------------------------------------------------------------- #
# Series → directory lookup
# --------------------------------------------------------------------------- #

def build_series_dir_map(root: str) -> dict[str, str]:
    """Map SeriesInstanceUID -> directory containing its DICOM files."""
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


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #

ASIR_COLORS = {"": "#2d6a4f", "30": "#52b788", "50": "#f4a261", "70": "#e63946"}
ASIR_LABELS = {"": "FBP (0%)", "30": "ASIR 30%", "50": "ASIR 50%", "70": "ASIR 70%"}
SIE_COLORS  = {"B20f": "#1d3557", "B20s": "#e63946", "B30f": "#457b9d", "B30s": "#f4a261"}
SIE_LABELS  = {"B20f": "B20f (FBP)", "B20s": "B20s (SAFIRE)", "B30f": "B30f (FBP)", "B30s": "B30s (SAFIRE)"}


def plot_nps(groups: dict, colors: dict, labels: dict, title: str, out_path: str):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for key, (freq, nps_list) in groups.items():
        arr = np.array(nps_list)
        med = np.median(arr, axis=0)
        lo  = np.percentile(arr, 25, axis=0)
        hi  = np.percentile(arr, 75, axis=0)
        c   = colors.get(key, "gray")
        lbl = labels.get(key, key)
        ax.plot(freq, med, color=c, lw=2, label=lbl)
        ax.fill_between(freq, lo, hi, color=c, alpha=0.15)
    ax.set_xlabel("Spatial frequency (cycles/pixel)", fontsize=11)
    ax.set_ylabel("NPS (HU² · pixel²)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_xlim(0, 0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_pca(feature_rows: list[dict], color_key: str, colors: dict,
             labels: dict, title: str, out_path: str):
    if len(feature_rows) < 2:
        print(f"  not enough series for PCA ({len(feature_rows)}), skipping")
        return
    keys = ["noise_std", "sharpness", "freq_ratio", "corr_ratio"]
    X = np.array([[r[k] for k in keys] for r in feature_rows])
    groups_col = [r[color_key] for r in feature_rows]
    Xs = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2)
    coords = pca.fit_transform(Xs)
    fig, ax = plt.subplots(figsize=(6, 5))
    seen = set()
    for i, g in enumerate(groups_col):
        c   = colors.get(g, "gray")
        lbl = labels.get(g, g) if g not in seen else "_nolegend_"
        seen.add(g)
        ax.scatter(coords[i, 0], coords[i, 1], color=c, label=lbl, s=60, alpha=0.8)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.0f}%)", fontsize=10)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.0f}%)", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inv",  required=True, help="series_inventory.csv from qiba_tag_inventory.py")
    ap.add_argument("--root", required=True, help="QIBA DICOM root directory")
    ap.add_argument("--out",  default="qiba_nps_out")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    inv = pd.read_csv(args.inv, dtype=str).fillna("")
    print(f"Loaded {len(inv)} series from inventory.")

    print("Building series->directory map (this takes a minute on first run)...")
    uid_map = build_series_dir_map(args.root)
    print(f"  mapped {len(uid_map)} series UIDs to directories.")

    # ---- select series -------------------------------------------------------
    ge_mask = (
        (inv["Manufacturer"].str.upper().str.contains("GE")) &
        (inv["ConvolutionKernel"] == GE_TARGET_KERNEL) &
        (inv["desc_mA"] == GE_TARGET_MA)
    )
    ge_inv = inv[ge_mask].copy()
    print(f"GE STANDARD 275mA: {len(ge_inv)} series selected")

    sie_mask = (
        (inv["Manufacturer"].str.upper().str.contains("SIEMENS")) &
        (inv["ConvolutionKernel"].isin(SIE_KERNELS)) &
        (inv["Exposure"] == SIE_TARGET_MAS)
    )
    sie_inv = inv[sie_mask].copy()
    print(f"Siemens B20/B30 f/s at 250mAs: {len(sie_inv)} series selected")

    # ---- process each series -------------------------------------------------
    nps_ge   = defaultdict(list)   # asir_pct -> [(freq, nps)]
    nps_sie  = defaultdict(list)   # kernel   -> [(freq, nps)]
    feat_rows = []
    nps_table = []

    def process(row, group_key, group_val, nps_store):
        uid = row["SeriesInstanceUID"]
        d   = uid_map.get(uid)
        if not d:
            print(f"  ! directory not found for UID {uid[:20]}...")
            return
        slices = load_series_sorted(d)
        if len(slices) < 4:
            print(f"  ! only {len(slices)} slices for {row['SeriesDescription'][:50]}, skipping")
            return
        rois = sample_rois(slices, N_SLICES, ROI_SIZE, N_ROIS, HU_MIN, HU_MAX)
        if len(rois) < 4:
            print(f"  ! too few ROIs for {row['SeriesDescription'][:50]}, skipping")
            return
        freq, nps = compute_nps_2d(rois)
        nps_store[group_val].append((freq, nps))

        feats = fingerprint_features(rois)
        feats[group_key] = group_val
        feats["desc"] = row["SeriesDescription"]
        feats["manufacturer"] = row["Manufacturer"]
        feats["kernel"] = row["ConvolutionKernel"]
        feats["asir"] = row.get("desc_asir_pct", "")
        feat_rows.append(feats)

        peak_idx = np.argmax(nps)
        nps_table.append({
            "series": row["SeriesDescription"],
            "manufacturer": row["Manufacturer"],
            "kernel": row["ConvolutionKernel"],
            "asir_pct": row.get("desc_asir_pct", ""),
            "nps_peak_freq": f"{freq[peak_idx]:.4f}",
            "nps_peak_val":  f"{nps[peak_idx]:.2f}",
            "nps_integral":  f"{np.trapz(nps, freq):.2f}",
        })
        print(f"  processed: {row['SeriesDescription'][:60]}")

    print("\n--- Processing GE series ---")
    for _, row in ge_inv.iterrows():
        process(row, "asir", row.get("desc_asir_pct", ""), nps_ge)

    print("\n--- Processing Siemens series ---")
    for _, row in sie_inv.iterrows():
        process(row, "kernel", row["ConvolutionKernel"], nps_sie)

    # ---- NPS plots -----------------------------------------------------------
    print("\n--- Generating figures ---")

    # convert to freq/nps_list structure
    ge_plot   = {k: (v[0][0], [x[1] for x in v]) for k, v in nps_ge.items() if v}
    sie_plot  = {k: (v[0][0], [x[1] for x in v]) for k, v in nps_sie.items() if v}

    if ge_plot:
        plot_nps(ge_plot, ASIR_COLORS, ASIR_LABELS,
                 "NPS by ASIR level — GE STANDARD kernel, 275 mA\n"
                 "(ConvolutionKernel = 'STANDARD' for all curves)",
                 os.path.join(args.out, "nps_curves_GE.png"))

    if sie_plot:
        plot_nps(sie_plot, SIE_COLORS, SIE_LABELS,
                 "NPS by reconstruction — Siemens 250 mAs\n"
                 "(B20f/B30f = FBP, B20s/B30s = SAFIRE)",
                 os.path.join(args.out, "nps_curves_Siemens.png"))

    # ---- PCA plots -----------------------------------------------------------
    ge_feats  = [r for r in feat_rows if "GE" in r.get("manufacturer", "").upper()]
    sie_feats = [r for r in feat_rows if "SIEMENS" in r.get("manufacturer", "").upper()]

    if ge_feats:
        plot_pca(ge_feats, "asir", ASIR_COLORS, ASIR_LABELS,
                 "Feature space — GE STANDARD kernel by ASIR level\n"
                 "(same ConvolutionKernel tag; pixel-derived separation)",
                 os.path.join(args.out, "feature_pca_GE.png"))

    if sie_feats:
        plot_pca(sie_feats, "kernel", SIE_COLORS, SIE_LABELS,
                 "Feature space — Siemens FBP vs SAFIRE",
                 os.path.join(args.out, "feature_pca_Siemens.png"))

    # ---- CSV outputs ---------------------------------------------------------
    if nps_table:
        pd.DataFrame(nps_table).to_csv(
            os.path.join(args.out, "nps_table.csv"), index=False)
        print(f"\nWrote nps_table.csv ({len(nps_table)} rows)")

    if feat_rows:
        pd.DataFrame(feat_rows).to_csv(
            os.path.join(args.out, "features.csv"), index=False)
        print(f"Wrote features.csv ({len(feat_rows)} rows)")

    print("\nDone. Check nps_curves_GE.png first — "
          "if the four curves separate, the thesis is proven.")


if __name__ == "__main__":
    main()
