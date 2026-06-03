"""QIBA IR Phantom — Model observer detectability analysis.

Computes the non-prewhitening matched filter (NPWMF) detectability index d'
for the 10 known spherical lesions across ASIR conditions.

Physical basis:
    d' = C * integral[ W(f) * H_signal(f) df ] / sqrt( integral[ NPS(f) * H_signal^2(f) df ] )

Simplified (ideal observer, circular disc signal):
    d' = C * sqrt(pi * R^2) / sigma_eff

where sigma_eff = sqrt( integral[ NPS(f) df ] ) is the effective noise
from the measured NPS, and R is the lesion radius in physical units.

ASIR reshapes the NPS — reducing low-frequency noise but potentially
preserving or even raising high-frequency components relative to FBP.
For large lesions (low-frequency signal), ASIR → lower sigma_eff → higher d'.
For small lesions (high-frequency signal), edge softening dominates → d' may fall.

This demonstrates size-dependent clinical consequence from reconstruction
method changes that are INVISIBLE to the ConvolutionKernel DICOM tag.

Known ground truth from QIBA IR phantom documentation:
  - 10 spherical lesions: 5 sizes (5, 7, 10, 14, 20mm), 2 HU targets (95, 110)
  - Liver parenchyma background: ~40-60 HU
  - Fatty parenchyma background: ~-20-20 HU (lesions on the border)

Outputs (--out dir):
  detectability_by_size.png     d' vs lesion size, one curve per ASIR level
  detectability_heatmap.png     d' as heatmap: size x condition
  nps_sigma_by_asir.png         effective noise sigma from NPS integral
  detectability.csv             full d' table

Usage:
  python validation/qiba_detectability.py \
      --inv  G:/GammaMetric/qiba_inventory_out/series_inventory.csv \
      --root G:/GammaMetric/manifest-1619103006849/QIBA-CT-Liver-Phantom \
      --out  G:/GammaMetric/qiba_detectability_out
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
from scipy.ndimage import sobel

# ------------------------------------------------------------------ #
# Known lesion parameters (QIBA IR phantom documentation)
# ------------------------------------------------------------------ #
LESION_DIAMETERS_MM = [5.0, 7.0, 10.0, 14.0, 20.0]
LESION_HU_TARGETS   = [95, 110]   # HU — lesion radiodensities
LIVER_BG_HU         = 50.0        # typical liver parenchyma in IR phantom

# ------------------------------------------------------------------ #
# NPS computation (same ROI approach as qiba_nps_analysis.py)
# ------------------------------------------------------------------ #
ROI_SIZE   = 64
N_ROIS     = 20
N_SLICES   = 12
BG_HU_MIN  = 20.0    # uniform liver parenchyma (not lesion, not fat)
BG_HU_MAX  = 80.0
BG_STD_MAX = 35.0    # tight: reject heterogeneous ROIs

# ------------------------------------------------------------------ #
# Series — 275 mA, 2.5 mm, GE STANDARD, one per ASIR level
# ------------------------------------------------------------------ #
ASIR_SERIES = {
    "FBP":      "1.2.840.113619.2.340.3.1930077730.508.1439460703.158.2",
    "ASIR 30%": "1.2.840.113619.2.340.3.1930077730.508.1439460703.158.5",
    "ASIR 50%": "1.2.840.113619.2.340.3.1930077730.508.1439460703.158.8",
    "ASIR 70%": "1.2.840.113619.2.340.3.1930077730.199.1439484877.308",
}  # 690 mA, 2.5 mm, GE STANDARD kernel
ASIR_ORDER  = ["FBP", "ASIR 30%", "ASIR 50%", "ASIR 70%"]
ASIR_COLORS = {
    "FBP":      "#2d6a4f",
    "ASIR 30%": "#52b788",
    "ASIR 50%": "#f4a261",
    "ASIR 70%": "#e63946",
}


# ------------------------------------------------------------------ #
# DICOM loading
# ------------------------------------------------------------------ #

def build_uid_map(root: str) -> dict[str, str]:
    uid_to_dir: dict[str, str] = {}
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


def load_series_slices(dicom_dir: str) -> tuple[list[np.ndarray], float]:
    """Return (sorted HU slices, pixel_spacing_mm)."""
    entries = []
    px = 1.0
    for fn in os.listdir(dicom_dir):
        p = os.path.join(dicom_dir, fn)
        try:
            ds = pydicom.dcmread(p, stop_before_pixels=False)
            if not hasattr(ds, "pixel_array"):
                continue
            pos = float(ds.ImagePositionPatient[2]) \
                  if hasattr(ds, "ImagePositionPatient") \
                  else float(getattr(ds, "InstanceNumber", 0))
            arr = ds.pixel_array.astype(np.float32)
            hu  = arr * float(getattr(ds, "RescaleSlope", 1)) \
                      + float(getattr(ds, "RescaleIntercept", 0))
            px  = float(ds.PixelSpacing[0]) if hasattr(ds, "PixelSpacing") else 1.0
            entries.append((pos, hu))
        except Exception:
            continue
    entries.sort(key=lambda x: x[0])
    return [e[1] for e in entries], px


# ------------------------------------------------------------------ #
# NPS measurement
# ------------------------------------------------------------------ #

def sample_bg_rois(slices: list[np.ndarray], n_slices: int,
                   roi_size: int, n_rois: int) -> np.ndarray:
    """Return (N, roi_size, roi_size) uniform liver-parenchyma ROIs."""
    mid = len(slices) // 2
    sel = slices[max(0, mid - n_slices // 2): mid + n_slices // 2]
    rng = np.random.default_rng(42)
    rois = []
    for sl in sel:
        H, W = sl.shape
        attempts, found = 0, 0
        while found < n_rois and attempts < n_rois * 30:
            r = rng.integers(0, H - roi_size)
            c = rng.integers(0, W - roi_size)
            patch = sl[r:r + roi_size, c:c + roi_size]
            mu, sg = patch.mean(), patch.std()
            if BG_HU_MIN < mu < BG_HU_MAX and sg < BG_STD_MAX:
                rois.append(patch)
                found += 1
            attempts += 1
    return np.array(rois) if rois else np.zeros((1, roi_size, roi_size))


def compute_nps(rois: np.ndarray, px_mm: float) -> tuple[np.ndarray, np.ndarray]:
    """Radially-averaged NPS. Returns (freq_mm-1, nps_HU2*mm2)."""
    n, H, W = rois.shape
    psds = []
    for roi in rois:
        ft = np.fft.fft2(roi - roi.mean())
        psd = (np.abs(ft) ** 2) * (px_mm ** 2) / (H * W)
        psds.append(np.fft.fftshift(psd))
    mean_psd = np.mean(psds, axis=0)
    cy, cx = H // 2, W // 2
    y, x = np.ogrid[:H, :W]
    r_map = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(int)
    max_r = min(cx, cy)
    nps = np.array([mean_psd[r_map == i].mean() if (r_map == i).any() else 0.0
                    for i in range(max_r)])
    freq = np.arange(max_r) / (H * px_mm)   # cycles/mm
    return freq, nps


def sigma_eff(freq: np.ndarray, nps: np.ndarray) -> float:
    """Effective noise = sqrt(2D-correct integral of NPS: integral NPS(f)*2*pi*f df).
    The 2*pi*f Jacobian gives the true 2D noise variance, suppressing the
    dominant very-low-f phantom-structure peak that inflates the naive 1D integral."""
    w = 2 * np.pi * freq
    return float(np.sqrt(np.trapz(nps * w, freq)))


# ------------------------------------------------------------------ #
# Detectability index
# ------------------------------------------------------------------ #

def disc_signal_spectrum(freq: np.ndarray, radius_mm: float) -> np.ndarray:
    """
    Fourier transform of a uniform disc (2D circularly symmetric):
        H(f) = pi * R^2 * 2 * J1(2*pi*f*R) / (2*pi*f*R)
    Normalised so H(0) = 1.
    Used as the signal template in the matched-filter observer.
    """
    from scipy.special import j1
    f = freq.copy()
    f[f == 0] = 1e-12   # avoid divide-by-zero
    arg = 2 * np.pi * f * radius_mm
    H = 2 * j1(arg) / arg
    H = np.abs(H)
    H[freq == 0] = 1.0
    return H


def detectability_index(freq: np.ndarray, nps: np.ndarray,
                         contrast_hu: float, radius_mm: float) -> float:
    """
    Non-prewhitening matched filter (NPWMF) detectability index,
    correct 2D radial form:

        d' = C * integral[ H^2(f) * 2*pi*f df ]
                 / sqrt( integral[ NPS(f) * H^2(f) * 2*pi*f df ] )

    The 2*pi*f factor is the Jacobian for converting the 2D polar
    frequency integral to a 1D radial integral (area element = 2*pi*f df).
    Without it the integral is over-weighted at low frequencies, where
    the very-low-f NPS peak from phantom structure dominates and masks
    the reconstruction-dependent mid-frequency noise differences.

    H(f) = Fourier transform of a uniform disc (Bessel-J1 function):
        H(f) = 2*J1(2*pi*f*R) / (2*pi*f*R)
    This falls off with frequency — large lesions concentrate signal at low f,
    small lesions extend into high f — so only the NPS at frequencies the
    lesion actually occupies counts toward the denominator.

    Reference: Burgess (1994) Med Phys; Barrett & Myers (2004) Foundations.
    """
    H  = disc_signal_spectrum(freq, radius_mm)
    H2 = H ** 2
    w  = 2 * np.pi * freq          # Jacobian: 2*pi*f
    nps_safe = np.where(nps > 0, nps, 1e-12)
    numerator   = np.trapz(H2 * w, freq)
    denominator = np.trapz(nps_safe * H2 * w, freq)
    if denominator <= 0:
        return 0.0
    return float(contrast_hu * numerator / np.sqrt(denominator))


# ------------------------------------------------------------------ #
# Plotting
# ------------------------------------------------------------------ #

def plot_sigma(sigma_dict: dict, out_path: str):
    conds = [c for c in ASIR_ORDER if c in sigma_dict]
    vals  = [sigma_dict[c] for c in conds]
    colors = [ASIR_COLORS[c] for c in conds]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(conds, vals, color=colors, edgecolor="white")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.05, f"{v:.2f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Effective noise sigma (HU)", fontsize=10)
    ax.set_title("Effective noise from NPS integral — GE STANDARD kernel\n"
                 "(ConvolutionKernel = 'STANDARD' for all bars)", fontsize=10)
    ax.tick_params(axis="x", rotation=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_detectability(d_prime: dict, out_path: str):
    """d_prime: {condition: {(diam_mm, hu_target): d_val}}"""
    conds = [c for c in ASIR_ORDER if c in d_prime]
    diams = sorted(LESION_DIAMETERS_MM)
    hu_targets = sorted(LESION_HU_TARGETS)

    fig, axes = plt.subplots(1, len(hu_targets),
                             figsize=(6 * len(hu_targets), 5), sharey=True)
    if len(hu_targets) == 1:
        axes = [axes]

    for ax, hu in zip(axes, hu_targets):
        contrast = hu - LIVER_BG_HU
        for cond in conds:
            vals = [d_prime[cond].get((d, hu), np.nan) for d in diams]
            ax.plot(diams, vals, marker="o", color=ASIR_COLORS[cond],
                    label=cond, lw=2)
        ax.axhline(1.0, color="gray", lw=1, ls="--", alpha=0.6)
        ax.axhline(3.0, color="black", lw=1, ls="--", alpha=0.4)
        ax.text(diams[-1] + 0.3, 1.05, "d'=1", fontsize=7, color="gray")
        ax.text(diams[-1] + 0.3, 3.05, "d'=3", fontsize=7, color="black")
        ax.set_xlabel("Lesion diameter (mm)", fontsize=10)
        ax.set_ylabel("Detectability index d'", fontsize=10)
        ax.set_title(f"Lesion HU={hu}  (contrast={contrast:.0f} HU above liver)\n"
                     f"ConvolutionKernel='STANDARD' for all curves", fontsize=9)
        ax.legend(fontsize=8)
        ax.set_xlim(3, 22)

    fig.suptitle("NPWMF Detectability index — QIBA IR Phantom\n"
                 "Size-dependent clinical consequence of ASIR reconstruction",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_heatmap(d_prime: dict, out_path: str):
    conds = [c for c in ASIR_ORDER if c in d_prime]
    diams = sorted(LESION_DIAMETERS_MM)
    hu_targets = sorted(LESION_HU_TARGETS)

    fig, axes = plt.subplots(1, len(hu_targets),
                             figsize=(5 * len(hu_targets), 3.5))
    if len(hu_targets) == 1:
        axes = [axes]

    for ax, hu in zip(axes, hu_targets):
        mat = np.array([[d_prime[c].get((d, hu), np.nan)
                         for d in diams] for c in conds])
        im = ax.imshow(mat, aspect="auto", cmap="RdYlGn",
                       vmin=0, vmax=mat[~np.isnan(mat)].max() * 1.1)
        ax.set_xticks(range(len(diams)))
        ax.set_xticklabels([f"{d}mm" for d in diams], fontsize=8)
        ax.set_yticks(range(len(conds)))
        ax.set_yticklabels(conds, fontsize=8)
        ax.set_title(f"d'  —  {hu} HU lesions", fontsize=9)
        plt.colorbar(im, ax=ax, label="d'")
        # annotate cells
        for i, c in enumerate(conds):
            for j, d in enumerate(diams):
                v = d_prime[c].get((d, hu), np.nan)
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                            fontsize=7,
                            color="white" if v < mat[~np.isnan(mat)].mean() else "black")

    fig.suptitle("Detectability heatmap (green=better)\n"
                 "Rows = ASIR level  |  Cols = lesion size",
                 fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inv",  required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out",  default="qiba_detectability_out")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    inv = pd.read_csv(args.inv, dtype=str).fillna("")
    print("Building UID->dir map...")
    uid_map = build_uid_map(args.root)
    print(f"  mapped {len(uid_map)} series")

    # ---- measure NPS per condition ------------------------------------------
    nps_data:   dict[str, tuple] = {}  # condition -> (freq, nps)
    sigma_data: dict[str, float] = {}

    for label, uid in ASIR_SERIES.items():
        d = uid_map.get(uid)
        if not d:
            print(f"  ! UID not found for {label}")
            continue
        slices, px = load_series_slices(d)
        print(f"  {label}: {len(slices)} slices, px={px:.3f}mm")
        rois = sample_bg_rois(slices, N_SLICES, ROI_SIZE, N_ROIS)
        print(f"    sampled {len(rois)} background ROIs")
        if len(rois) < 4:
            print(f"    ! too few ROIs, skipping")
            continue
        freq, nps = compute_nps(rois, px)
        sig = sigma_eff(freq, nps)
        nps_data[label]   = (freq, nps, px)
        sigma_data[label] = sig
        print(f"    sigma_eff = {sig:.3f} HU")

    if not nps_data:
        print("No series processed. Exiting.")
        return

    # ---- compute d' for every (condition, lesion_size, lesion_hu) -----------
    d_prime: dict[str, dict] = {c: {} for c in nps_data}
    rows = []

    for label, (freq, nps, px) in nps_data.items():
        for diam_mm in LESION_DIAMETERS_MM:
            radius_mm = diam_mm / 2.0
            for hu_target in LESION_HU_TARGETS:
                contrast = hu_target - LIVER_BG_HU
                dp = detectability_index(freq, nps, contrast, radius_mm)
                d_prime[label][(diam_mm, hu_target)] = dp
                rows.append({
                    "condition":    label,
                    "diam_mm":      diam_mm,
                    "hu_target":    hu_target,
                    "contrast_hu":  contrast,
                    "d_prime":      round(dp, 4),
                    "sigma_eff":    round(sigma_data[label], 4),
                })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(args.out, "detectability.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nWrote {csv_path}")
    print("\n--- d' summary by condition and size ---")
    pivot = df.pivot_table(index="condition", columns="diam_mm",
                           values="d_prime", aggfunc="mean")
    print(pivot.to_string())

    # ---- figures -------------------------------------------------------------
    print("\n--- Generating figures ---")
    plot_sigma(sigma_data,
               os.path.join(args.out, "nps_sigma_by_asir.png"))
    plot_detectability(d_prime,
                       os.path.join(args.out, "detectability_by_size.png"))
    plot_heatmap(d_prime,
                 os.path.join(args.out, "detectability_heatmap.png"))

    # ---- crossover analysis --------------------------------------------------
    fbp_dp = d_prime.get("FBP", {})
    print("\n--- Crossover analysis: where does ASIR 70% help vs hurt? ---")
    for diam_mm in LESION_DIAMETERS_MM:
        for hu_target in LESION_HU_TARGETS:
            fbp_val  = fbp_dp.get((diam_mm, hu_target), np.nan)
            asir_val = d_prime.get("ASIR 70%", {}).get((diam_mm, hu_target), np.nan)
            if np.isnan(fbp_val) or np.isnan(asir_val):
                continue
            delta = asir_val - fbp_val
            direction = "BETTER" if delta > 0 else "WORSE "
            print(f"  {diam_mm:4.0f}mm / {hu_target}HU:  "
                  f"FBP={fbp_val:.2f}  ASIR70%={asir_val:.2f}  "
                  f"delta={delta:+.2f}  [{direction}]")

    print("\nDone.")


if __name__ == "__main__":
    main()
