"""QIBA liver phantom — insert CNR and radiomics feature analysis.

Demonstrates clinical consequence of ASIR/iterative reconstruction on
liver lesion simulants with known ground-truth positions.

Pipeline:
  1. Auto-detect insert positions on high-dose FBP reference (690 mA, 1.25mm)
     using blob_log, with interactive verification figure saved for inspection.
  2. For each ASIR condition (FBP / 30 / 50 / 70%) at matched dose (275 mA):
     - Extract insert ROIs and annular background ROIs (same positions)
     - Compute CNR = |mu_insert - mu_bg| / sigma_bg  per insert
     - Extract radiomics features: histogram stats + GLCM texture
  3. Output:
     inserts_detected.png       blob-detected inserts on FBP reference
     cnr_by_asir.png            CNR per insert per ASIR level (bar + line)
     feature_drift.png          L2 feature distance from FBP reference
     insert_cnr.csv             per-insert per-condition CNR table
     insert_features.csv        full feature vectors

Usage:
  python validation/qiba_insert_analysis.py \
      --inv  G:/GammaMetric/qiba_inventory_out/series_inventory.csv \
      --root G:/GammaMetric/manifest-1619103006849/QIBA-CT-Liver-Phantom \
      --out  G:/GammaMetric/qiba_insert_out
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
import matplotlib.patches as mpatches
from scipy.ndimage import label as ndlabel
from skimage.feature import blob_log
from skimage.exposure import rescale_intensity
from skimage.feature import graycomatrix, graycoprops

# ------------------------------------------------------------------ #
# Series UIDs — 275 mA, 2.5 mm, GE STANDARD, one per ASIR level
# (from qiba_tag_inventory output)
# ------------------------------------------------------------------ #
ASIR_SERIES = {
    "FBP":      "1.2.840.113619.2.340.3.1930077730.508.1439460703.218.12",
    "ASIR 30%": "1.2.840.113619.2.340.3.1930077730.508.1439460703.218.15",
    "ASIR 50%": "1.2.840.113619.2.340.3.1930077730.508.1439460703.218.18",
    "ASIR 70%": "1.2.840.113619.2.340.3.1930077730.199.1439484877.320",
}

# High-dose FBP reference for insert detection (best SNR)
REF_SERIES_DESC_PATTERN = "690 mA 1.25 MM"   # fallback pattern if UID unknown
REF_ASIR_KEY = "FBP"                           # use FBP arm at 275mA if no 690 found

# Blob detection parameters (tune if inserts are missed/false-positives)
BLOB_MIN_SIGMA  = 3    # 5mm lesion @ 0.781mm/px = 3.2px sigma
BLOB_MAX_SIGMA  = 17   # 20mm lesion = 12.8px radius → sigma ~9
BLOB_THRESHOLD  = 0.04
BLOB_OVERLAP    = 0.4

# ROI geometry
INSERT_RADIUS_PX  = 8    # conservative: smaller than smallest lesion (5mm=3.2px)
BG_INNER_PX       = 14   # annular background inner radius
BG_OUTER_PX       = 22   # annular background outer radius

# Dataset #3 (IR phantom) known lesion HU: 95 and 110 HU
# Liver background ~40–60 HU; detect lesions in this window
INSERT_HU_MIN = 75    # below 95 HU lesion target (some noise)
INSERT_HU_MAX = 130   # above 110 HU lesion target (some noise)

# GLCM parameters
GLCM_DISTANCES  = [1, 2]
GLCM_ANGLES     = [0, np.pi/4, np.pi/2, 3*np.pi/4]
GLCM_LEVELS     = 64
GLCM_PROPS      = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]


# ------------------------------------------------------------------ #
# DICOM utilities
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


def load_mid_slice(dicom_dir: str) -> tuple[np.ndarray, float]:
    """Return (HU array, pixel_spacing_mm) for the slice with most lesion-HU voxels.
    The IR phantom inserts (95–110 HU) are on the fatty/normal parenchyma border,
    which may not be the axial midpoint — pick the slice richest in insert-HU voxels."""
    entries = []
    for fn in os.listdir(dicom_dir):
        p = os.path.join(dicom_dir, fn)
        try:
            ds = pydicom.dcmread(p, stop_before_pixels=False)
            if not hasattr(ds, "pixel_array"):
                continue
            pos = float(ds.ImagePositionPatient[2]) \
                  if hasattr(ds, "ImagePositionPatient") \
                  else float(getattr(ds, "InstanceNumber", 0))
            entries.append((pos, ds))
        except Exception:
            continue
    if not entries:
        raise ValueError(f"No DICOM in {dicom_dir}")
    entries.sort(key=lambda x: x[0])
    px = float(entries[0][1].PixelSpacing[0]) \
         if hasattr(entries[0][1], "PixelSpacing") else 1.0

    def to_hu(ds):
        arr = ds.pixel_array.astype(np.float32)
        return arr * float(getattr(ds, "RescaleSlope", 1)) + float(getattr(ds, "RescaleIntercept", 0))

    # Score each slice: count pixels in lesion-HU range (75–130 HU) inside body
    best_score, best_hu = -1, None
    for _, ds in entries:
        hu = to_hu(ds)
        score = int(((hu > INSERT_HU_MIN) & (hu < INSERT_HU_MAX) & (hu > -200)).sum())
        if score > best_score:
            best_score = score
            best_hu = hu
    return best_hu, px


# ------------------------------------------------------------------ #
# Insert detection
# ------------------------------------------------------------------ #

def detect_inserts(hu: np.ndarray) -> list[tuple[int, int, float]]:
    """
    Detect IR-phantom lesion simulants using known HU targets (95, 110 HU).
    Lesions are on the fatty/normal liver parenchyma border.
    Strategy:
      1. Build liver mask (body interior, 20–90 HU — liver parenchyma range)
      2. Find lesion-HU regions (INSERT_HU_MIN–INSERT_HU_MAX) within or near liver
      3. Run blob_log on the lesion-enhanced map to localise centres
    Returns list of (row, col, radius_px).
    """
    H, W = hu.shape
    body_mask   = hu > -200    # exclude air
    # liver parenchyma window: normal liver ~40–70 HU, fatty ~-30–20 HU
    liver_mask  = body_mask & (hu > 20) & (hu < 90)
    # lesion-contrast map: how much above liver background
    liver_mean  = hu[liver_mask].mean() if liver_mask.any() else 50.0
    # enhance regions at lesion HU (95–110) relative to liver bg
    lesion_map  = np.where(
        body_mask & (hu > INSERT_HU_MIN) & (hu < INSERT_HU_MAX),
        hu - liver_mean,
        0.0
    ).astype(np.float32)
    if lesion_map.max() < 1:
        return []
    norm = rescale_intensity(lesion_map, out_range=(0.0, 1.0))

    blobs = blob_log(norm,
                     min_sigma=BLOB_MIN_SIGMA,
                     max_sigma=BLOB_MAX_SIGMA,
                     num_sigma=12,
                     threshold=BLOB_THRESHOLD,
                     overlap=BLOB_OVERLAP)
    results = []
    cy, cx  = H // 2, W // 2
    max_dist = min(H, W) * 0.42
    margin   = int(BG_OUTER_PX + 4)
    seen_positions = []  # deduplicate very close blobs
    for r, c, sigma in blobs:
        radius = sigma * np.sqrt(2)
        # must be inside body, not too peripheral (not the phantom shell)
        dist = np.sqrt((r - cy)**2 + (c - cx)**2)
        if dist > max_dist:
            continue
        if not (margin < r < H - margin and margin < c < W - margin):
            continue
        # confirm insert HU in central ROI
        r_i, c_i = int(r), int(c)
        ins_mask = circular_mask(hu.shape, r_i, c_i, max(3, int(radius * 0.6)))
        if ins_mask.sum() < 5:
            continue
        mu = float(hu[ins_mask].mean())
        if not (INSERT_HU_MIN - 10 < mu < INSERT_HU_MAX + 10):
            continue
        # deduplicate: skip if too close to an already accepted blob
        too_close = any(np.sqrt((r_i - pr)**2 + (c_i - pc)**2) < radius * 1.5
                        for pr, pc in seen_positions)
        if too_close:
            continue
        seen_positions.append((r_i, c_i))
        results.append((r_i, c_i, float(radius)))
    return results


def circular_mask(shape, cy, cx, r_inner, r_outer=None):
    """Boolean mask for a disk (r_outer=None) or annulus."""
    Y, X = np.ogrid[:shape[0], :shape[1]]
    dist2 = (Y - cy) ** 2 + (X - cx) ** 2
    if r_outer is None:
        return dist2 <= r_inner ** 2
    return (dist2 >= r_inner ** 2) & (dist2 <= r_outer ** 2)


# ------------------------------------------------------------------ #
# CNR
# ------------------------------------------------------------------ #

def compute_cnr(hu: np.ndarray, inserts: list) -> list[dict]:
    rows = []
    for i, (r, c, _rad) in enumerate(inserts):
        ins_mask = circular_mask(hu.shape, r, c, INSERT_RADIUS_PX)
        bg_mask  = circular_mask(hu.shape, r, c, BG_INNER_PX, BG_OUTER_PX)
        mu_ins = hu[ins_mask].mean()
        mu_bg  = hu[bg_mask].mean()
        sg_bg  = hu[bg_mask].std()
        cnr    = abs(mu_ins - mu_bg) / (sg_bg + 1e-9)
        rows.append({
            "insert_id": i,
            "row": r, "col": c,
            "mu_insert": round(float(mu_ins), 2),
            "mu_bg":     round(float(mu_bg),  2),
            "sigma_bg":  round(float(sg_bg),  2),
            "cnr":       round(float(cnr),     3),
        })
    return rows


# ------------------------------------------------------------------ #
# Radiomics features (histogram + GLCM)
# ------------------------------------------------------------------ #

def extract_features(hu: np.ndarray, inserts: list) -> list[dict]:
    rows = []
    for i, (r, c, _rad) in enumerate(inserts):
        ins_mask = circular_mask(hu.shape, r, c, INSERT_RADIUS_PX)
        patch = hu[ins_mask]
        if patch.size < 9:
            continue

        # --- histogram features ---
        feats: dict = {"insert_id": i}
        feats["mean_hu"]    = float(patch.mean())
        feats["std_hu"]     = float(patch.std())
        feats["p10_hu"]     = float(np.percentile(patch, 10))
        feats["p90_hu"]     = float(np.percentile(patch, 90))
        feats["iqr_hu"]     = float(np.percentile(patch, 75) - np.percentile(patch, 25))
        feats["skewness"]   = float(_skew(patch))
        feats["kurtosis"]   = float(_kurt(patch))

        # --- GLCM texture (on square bounding box) ---
        r0 = max(0, r - INSERT_RADIUS_PX)
        r1 = min(hu.shape[0], r + INSERT_RADIUS_PX + 1)
        c0 = max(0, c - INSERT_RADIUS_PX)
        c1 = min(hu.shape[1], c + INSERT_RADIUS_PX + 1)
        box = hu[r0:r1, c0:c1].copy()
        # quantise to GLCM_LEVELS bins
        lo, hi = np.percentile(box, 1), np.percentile(box, 99)
        if hi > lo:
            box_q = np.clip(
                ((box - lo) / (hi - lo) * (GLCM_LEVELS - 1)).astype(np.uint8),
                0, GLCM_LEVELS - 1)
            gcm = graycomatrix(box_q, distances=GLCM_DISTANCES,
                               angles=GLCM_ANGLES, levels=GLCM_LEVELS,
                               symmetric=True, normed=True)
            for prop in GLCM_PROPS:
                feats[f"glcm_{prop}"] = float(graycoprops(gcm, prop).mean())

        rows.append(feats)
    return rows


def _skew(x):
    x = x - x.mean()
    s = x.std()
    return float((x**3).mean() / (s**3 + 1e-9))


def _kurt(x):
    x = x - x.mean()
    s = x.std()
    return float((x**4).mean() / (s**4 + 1e-9)) - 3.0


# ------------------------------------------------------------------ #
# Plotting
# ------------------------------------------------------------------ #

ASIR_COLORS = {
    "FBP":      "#2d6a4f",
    "ASIR 30%": "#52b788",
    "ASIR 50%": "#f4a261",
    "ASIR 70%": "#e63946",
}
ASIR_ORDER = ["FBP", "ASIR 30%", "ASIR 50%", "ASIR 70%"]


def plot_inserts(hu, inserts, out_path):
    fig, ax = plt.subplots(figsize=(6, 6))
    lo, hi = np.percentile(hu, 1), np.percentile(hu, 99)
    ax.imshow(np.clip(hu, lo, hi), cmap="gray", vmin=lo, vmax=hi)
    for i, (r, c, rad) in enumerate(inserts):
        circ = plt.Circle((c, r), INSERT_RADIUS_PX, color="cyan",
                           fill=False, lw=1.5)
        ax.add_patch(circ)
        ax.text(c + INSERT_RADIUS_PX + 2, r, str(i), color="cyan", fontsize=8)
    ax.set_title(f"Auto-detected inserts (n={len(inserts)})\nFBP 275 mA 2.5 mm reference",
                 fontsize=10)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_cnr(cnr_data: dict, out_path: str):
    """cnr_data: {condition: [{insert_id, cnr, ...}]}"""
    conditions = [k for k in ASIR_ORDER if k in cnr_data]
    insert_ids = sorted({r["insert_id"] for rows in cnr_data.values() for r in rows})
    n_ins = len(insert_ids)
    if not n_ins:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Bar chart: mean CNR per condition
    mean_cnr = [np.mean([r["cnr"] for r in cnr_data[c]]) for c in conditions]
    std_cnr  = [np.std( [r["cnr"] for r in cnr_data[c]]) for c in conditions]
    colors = [ASIR_COLORS[c] for c in conditions]
    bars = ax1.bar(conditions, mean_cnr, color=colors, yerr=std_cnr,
                   capsize=5, edgecolor="white", linewidth=0.5)
    ax1.set_ylabel("CNR (mean ± SD across inserts)", fontsize=10)
    ax1.set_title("Mean lesion CNR by reconstruction\n"
                  "(ConvolutionKernel = 'STANDARD' for all)", fontsize=10)
    ax1.tick_params(axis="x", rotation=15)
    for bar, val in zip(bars, mean_cnr):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.1, f"{val:.2f}",
                 ha="center", va="bottom", fontsize=8)

    # Line chart: per-insert CNR across conditions
    cmap = plt.cm.get_cmap("tab10", n_ins)
    for j, ins_id in enumerate(insert_ids):
        cnrs = []
        for c in conditions:
            row = next((r for r in cnr_data[c] if r["insert_id"] == ins_id), None)
            cnrs.append(row["cnr"] if row else np.nan)
        ax2.plot(conditions, cnrs, marker="o", color=cmap(j),
                 label=f"Insert {ins_id}", alpha=0.8)
    ax2.set_ylabel("CNR", fontsize=10)
    ax2.set_title("Per-insert CNR across ASIR levels", fontsize=10)
    ax2.tick_params(axis="x", rotation=15)
    ax2.legend(fontsize=7, ncol=2)
    ax2.axhline(5, color="black", lw=1, ls="--", alpha=0.4,
                label="Rose criterion (CNR=5)")

    fig.suptitle("QIBA Liver Phantom — CNR Analysis\n"
                 "Same ConvolutionKernel tag, different iterative recon strength",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_feature_drift(feat_data: dict, out_path: str):
    """feat_data: {condition: [{insert_id, feature_cols...}]}"""
    conditions = [k for k in ASIR_ORDER if k in feat_data]
    if "FBP" not in feat_data or len(conditions) < 2:
        print("  not enough conditions for drift plot")
        return

    exclude = {"insert_id", "condition"}
    feat_cols = [k for k in feat_data["FBP"][0].keys()
                 if k not in exclude and not k.startswith("_")
                 and isinstance(feat_data["FBP"][0][k], (int, float))]

    # Compute per-insert L2 distance from FBP reference
    insert_ids = sorted({r["insert_id"] for rows in feat_data.values() for r in rows})
    drift_rows = []
    for ins_id in insert_ids:
        ref = next((r for r in feat_data["FBP"] if r["insert_id"] == ins_id), None)
        if not ref:
            continue
        ref_vec = np.array([ref.get(k, 0) for k in feat_cols], dtype=float)
        for cond in conditions:
            row = next((r for r in feat_data[cond] if r["insert_id"] == ins_id), None)
            if not row:
                continue
            vec = np.array([row.get(k, 0) for k in feat_cols], dtype=float)
            # Normalised L2 distance
            denom = np.linalg.norm(ref_vec) + 1e-9
            dist = np.linalg.norm(vec - ref_vec) / denom
            drift_rows.append({"condition": cond, "insert_id": ins_id, "drift": dist})

    df = pd.DataFrame(drift_rows)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for ins_id in insert_ids:
        sub = df[df["insert_id"] == ins_id]
        conds_plot = [c for c in conditions if c in sub["condition"].values]
        drifts = [sub[sub["condition"] == c]["drift"].values[0]
                  if c in sub["condition"].values else np.nan
                  for c in conds_plot]
        ax.plot(conds_plot, drifts, marker="o", alpha=0.7, label=f"Insert {ins_id}")

    # mean drift line
    mean_drift = [df[df["condition"] == c]["drift"].mean() for c in conditions]
    ax.plot(conditions, mean_drift, color="black", lw=2.5,
            marker="D", zorder=5, label="Mean")

    ax.set_ylabel("Normalised feature distance from FBP", fontsize=10)
    ax.set_title("Radiomic feature drift — ASIR vs FBP reference\n"
                 "Per-insert L2 distance in histogram + GLCM feature space",
                 fontsize=10)
    ax.tick_params(axis="x", rotation=15)
    ax.legend(fontsize=7, ncol=2)
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
    ap.add_argument("--out",  default="qiba_insert_out")
    ap.add_argument("--blob-threshold", type=float, default=BLOB_THRESHOLD,
                    help="Sensitivity for blob detection (lower = more blobs)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    inv = pd.read_csv(args.inv, dtype=str).fillna("")
    print("Building UID->dir map...")
    uid_map = build_uid_map(args.root)
    print(f"  mapped {len(uid_map)} series")

    # ---- Load all ASIR series -----------------------------------------------
    series_hu: dict[str, np.ndarray] = {}
    series_px: dict[str, float] = {}
    for label, uid in ASIR_SERIES.items():
        d = uid_map.get(uid)
        if not d:
            print(f"  ! UID not found for {label}, skipping")
            continue
        hu, px = load_mid_slice(d)
        series_hu[label] = hu
        series_px[label] = px
        print(f"  loaded {label}: shape={hu.shape}, px={px:.3f}mm, "
              f"HU {hu.min():.0f}..{hu.max():.0f}")

    if "FBP" not in series_hu:
        print("ERROR: FBP series not loaded. Cannot proceed.")
        return

    # ---- Detect inserts on FBP reference ------------------------------------
    print("\nDetecting inserts on FBP reference...")
    inserts = detect_inserts(series_hu["FBP"])
    print(f"  found {len(inserts)} candidate inserts")
    if not inserts:
        print("  No inserts found. Try lowering --blob-threshold (current "
              f"{args.blob_threshold}). Exiting.")
        return

    plot_inserts(series_hu["FBP"], inserts,
                 os.path.join(args.out, "inserts_detected.png"))

    # ---- CNR per condition --------------------------------------------------
    print("\nComputing CNR...")
    cnr_data: dict[str, list] = {}
    all_cnr_rows = []
    for label, hu in series_hu.items():
        rows = compute_cnr(hu, inserts)
        cnr_data[label] = rows
        for r in rows:
            r["condition"] = label
            all_cnr_rows.append(r)

    pd.DataFrame(all_cnr_rows).to_csv(
        os.path.join(args.out, "insert_cnr.csv"), index=False)
    print(f"  wrote insert_cnr.csv ({len(all_cnr_rows)} rows)")
    plot_cnr(cnr_data, os.path.join(args.out, "cnr_by_asir.png"))

    # ---- Radiomics features per condition -----------------------------------
    print("\nExtracting radiomics features...")
    feat_data: dict[str, list] = {}
    all_feat_rows = []
    for label, hu in series_hu.items():
        rows = extract_features(hu, inserts)
        feat_data[label] = rows
        for r in rows:
            r["condition"] = label
            all_feat_rows.append(r)

    pd.DataFrame(all_feat_rows).to_csv(
        os.path.join(args.out, "insert_features.csv"), index=False)
    print(f"  wrote insert_features.csv ({len(all_feat_rows)} rows)")
    plot_feature_drift(feat_data, os.path.join(args.out, "feature_drift.png"))

    # ---- Summary to console -------------------------------------------------
    print("\n--- CNR summary ---")
    df_cnr = pd.DataFrame(all_cnr_rows)
    summary = df_cnr.groupby("condition")["cnr"].agg(["mean", "std", "min", "max"])
    print(summary.to_string())

    print("\nDone. Check inserts_detected.png first to verify blob detection quality.")


if __name__ == "__main__":
    main()
