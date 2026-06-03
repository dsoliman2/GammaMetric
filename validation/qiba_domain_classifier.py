"""QIBA IR Phantom — FBP vs ASIR domain classifier.

Demonstrates that any AI trained on FBP images encounters a measurable
input distribution shift when applied to ASIR-reconstructed images of the
same phantom — even with identical ConvolutionKernel DICOM tags.

Approach:
  - Extract patches from phantom body across all ASIR conditions (FBP / 30 / 50 / 70%)
  - Compute the same 4 patch features as acquisition_fingerprint.py
  - Train Random Forest classifiers (FBP vs each ASIR level) with 5-fold CV
  - Show graded separability: harder to distinguish FBP from ASIR 30%,
    easier from ASIR 70% — matching the NPS separation already shown

The argument:
  If a simple 4-feature RF achieves >90% accuracy distinguishing FBP from ASIR 70%,
  then any AI that has learned FBP texture statistics will encounter an equivalent
  distribution shift on ASIR images — and the ConvolutionKernel tag provides
  no signal that this shift has occurred.

Outputs (--out dir):
  accuracy_by_asir.png      classification accuracy FBP vs each ASIR level
  roc_curves.png            ROC curves, one per comparison
  feature_importance.png    RF feature importance across comparisons
  confusion_matrices.png    per-comparison confusion matrices
  domain_classifier.csv     full results table

Usage:
  python validation/qiba_domain_classifier.py \
      --inv  G:/GammaMetric/qiba_inventory_out/series_inventory.csv \
      --root G:/GammaMetric/manifest-1619103006849/QIBA-CT-Liver-Phantom \
      --out  G:/GammaMetric/qiba_domain_out
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (accuracy_score, roc_auc_score,
                              roc_curve, confusion_matrix)
from sklearn.preprocessing import label_binarize

# ------------------------------------------------------------------ #
# Series — 690 mA, 2.5 mm, GE STANDARD
# ------------------------------------------------------------------ #
ASIR_SERIES = {
    "FBP":      "1.2.840.113619.2.340.3.1930077730.508.1439460703.158.2",
    "ASIR 30%": "1.2.840.113619.2.340.3.1930077730.508.1439460703.158.5",
    "ASIR 50%": "1.2.840.113619.2.340.3.1930077730.508.1439460703.158.8",
    "ASIR 70%": "1.2.840.113619.2.340.3.1930077730.199.1439484877.308",
}
ASIR_ORDER  = ["FBP", "ASIR 30%", "ASIR 50%", "ASIR 70%"]
ASIR_COLORS = {
    "FBP":      "#2d6a4f",
    "ASIR 30%": "#52b788",
    "ASIR 50%": "#f4a261",
    "ASIR 70%": "#e63946",
}

# Patch extraction
PATCH_SIZE   = 32    # pixels
N_PATCHES    = 300   # per slice per condition
N_SLICES     = 20    # central slices
HU_BODY_MIN  = -200  # exclude air
HU_PATCH_MAX_STD = 80  # exclude highly heterogeneous patches (edges, inserts)
RF_N_TREES   = 200
CV_FOLDS     = 5
RNG_SEED     = 42


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


def load_central_slices(dicom_dir: str, n_slices: int) -> list[np.ndarray]:
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
            arr = ds.pixel_array.astype(np.float32)
            hu  = arr * float(getattr(ds, "RescaleSlope", 1)) \
                      + float(getattr(ds, "RescaleIntercept", 0))
            entries.append((pos, hu))
        except Exception:
            continue
    entries.sort(key=lambda x: x[0])
    mid = len(entries) // 2
    sel = entries[max(0, mid - n_slices // 2): mid + n_slices // 2]
    return [e[1] for e in sel]


# ------------------------------------------------------------------ #
# Patch features — same as acquisition_fingerprint.py
# ------------------------------------------------------------------ #

def patch_features(patch: np.ndarray) -> np.ndarray:
    """4 features: noise_std, sharpness, freq_ratio, corr_ratio."""
    noise_std = patch.std()
    gx = sobel(patch, axis=0)
    gy = sobel(patch, axis=1)
    sharpness = np.sqrt(gx**2 + gy**2).mean()
    ft = np.abs(np.fft.fft2(patch))
    H = patch.shape[0]
    half = H // 2
    low  = ft[:half // 2, :half // 2].mean() + 1e-9
    high = ft[half // 2:, half // 2:].mean() + 1e-9
    freq_ratio = high / low
    flat = patch.flatten()
    corr_ratio = float(np.corrcoef(flat[:-1], flat[1:])[0, 1]) if len(flat) > 1 else 0.0
    return np.array([noise_std, sharpness, freq_ratio, corr_ratio])


FEATURE_NAMES = ["noise_std", "sharpness", "freq_ratio", "corr_ratio"]


def extract_patches(slices: list[np.ndarray], n_patches: int) -> np.ndarray:
    """Extract n_patches random body-interior patches per slice, return (N, 4) features."""
    rng = np.random.default_rng(RNG_SEED)
    rows = []
    for sl in slices:
        H, W = sl.shape
        if H < PATCH_SIZE or W < PATCH_SIZE:
            continue
        attempts, found = 0, 0
        while found < n_patches and attempts < n_patches * 15:
            r = rng.integers(0, H - PATCH_SIZE)
            c = rng.integers(0, W - PATCH_SIZE)
            patch = sl[r:r + PATCH_SIZE, c:c + PATCH_SIZE]
            # exclude air and highly heterogeneous regions (organ boundaries, inserts)
            if patch.mean() > HU_BODY_MIN and patch.std() < HU_PATCH_MAX_STD:
                rows.append(patch_features(patch))
                found += 1
            attempts += 1
    return np.array(rows) if rows else np.zeros((1, 4))


# ------------------------------------------------------------------ #
# Plotting
# ------------------------------------------------------------------ #

def plot_accuracy(results: list[dict], out_path: str):
    comparisons = [r["comparison"] for r in results]
    accs = [r["accuracy"] for r in results]
    aucs = [r["auc"] for r in results]
    colors = [ASIR_COLORS.get(r["asir_label"], "gray") for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    bars = ax1.bar(comparisons, accs, color=colors, edgecolor="white")
    ax1.axhline(0.5, color="gray", ls="--", lw=1, alpha=0.5, label="Chance")
    ax1.set_ylim(0.4, 1.05)
    ax1.set_ylabel("Classification accuracy (5-fold CV)", fontsize=10)
    ax1.set_title("FBP vs ASIR — patch-level domain separability\n"
                  "(ConvolutionKernel = 'STANDARD' for all)", fontsize=9)
    ax1.tick_params(axis="x", rotation=10)
    for bar, v in zip(bars, accs):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.005, f"{v:.3f}",
                 ha="center", va="bottom", fontsize=9)

    bars2 = ax2.bar(comparisons, aucs, color=colors, edgecolor="white")
    ax2.axhline(0.5, color="gray", ls="--", lw=1, alpha=0.5)
    ax2.set_ylim(0.4, 1.05)
    ax2.set_ylabel("ROC-AUC", fontsize=10)
    ax2.set_title("AUC — FBP vs ASIR discrimination", fontsize=9)
    ax2.tick_params(axis="x", rotation=10)
    for bar, v in zip(bars2, aucs):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.005, f"{v:.3f}",
                 ha="center", va="bottom", fontsize=9)

    fig.suptitle("Domain classifier: same phantom, same ConvolutionKernel tag, "
                 "different reconstruction\nRF trained on patch-level features",
                 fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_roc(results: list[dict], out_path: str):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4, label="Chance")
    for r in results:
        fpr, tpr = r["fpr"], r["tpr"]
        auc = r["auc"]
        c = ASIR_COLORS.get(r["asir_label"], "gray")
        ax.plot(fpr, tpr, color=c, lw=2,
                label=f"{r['comparison']} (AUC={auc:.3f})")
    ax.set_xlabel("False positive rate", fontsize=10)
    ax.set_ylabel("True positive rate", fontsize=10)
    ax.set_title("ROC — FBP vs ASIR domain classification\n"
                 "Patch features: noise_std, sharpness, freq_ratio, corr_ratio",
                 fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_feature_importance(results: list[dict], out_path: str):
    fig, axes = plt.subplots(1, len(results), figsize=(4 * len(results), 4))
    if len(results) == 1:
        axes = [axes]
    for ax, r in zip(axes, results):
        imp = r["feature_importance"]
        c = ASIR_COLORS.get(r["asir_label"], "gray")
        bars = ax.barh(FEATURE_NAMES, imp, color=c, edgecolor="white")
        ax.set_xlim(0, max(imp) * 1.3)
        ax.set_title(r["comparison"], fontsize=9)
        ax.set_xlabel("Importance", fontsize=8)
        for bar, v in zip(bars, imp):
            ax.text(v + 0.005, bar.get_y() + bar.get_height() / 2,
                    f"{v:.3f}", va="center", fontsize=7)
    fig.suptitle("RF feature importance — what drives FBP vs ASIR discrimination",
                 fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_confusion(results: list[dict], out_path: str):
    fig, axes = plt.subplots(1, len(results), figsize=(3.5 * len(results), 3.5))
    if len(results) == 1:
        axes = [axes]
    for ax, r in zip(axes, results):
        cm = r["confusion_matrix"]
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred FBP", f"Pred ASIR"], fontsize=8)
        ax.set_yticklabels(["True FBP", f"True ASIR"], fontsize=8)
        ax.set_title(r["comparison"], fontsize=9)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm_norm[i,j]:.2f}\n(n={cm[i,j]})",
                        ha="center", va="center", fontsize=8,
                        color="white" if cm_norm[i, j] > 0.6 else "black")
    fig.suptitle("Confusion matrices — 5-fold CV predictions",
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
    ap.add_argument("--out",  default="qiba_domain_out")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    inv = pd.read_csv(args.inv, dtype=str).fillna("")
    print("Building UID->dir map...")
    uid_map = build_uid_map(args.root)

    # ---- extract features per condition -------------------------------------
    condition_features: dict[str, np.ndarray] = {}
    for label, uid in ASIR_SERIES.items():
        d = uid_map.get(uid)
        if not d:
            print(f"  ! UID not found for {label}")
            continue
        slices = load_central_slices(d, N_SLICES)
        print(f"  {label}: {len(slices)} slices")
        feats = extract_patches(slices, N_PATCHES)
        condition_features[label] = feats
        print(f"    extracted {len(feats)} patches, "
              f"noise_std mean={feats[:,0].mean():.2f}")

    if "FBP" not in condition_features:
        print("FBP not loaded. Exiting.")
        return

    fbp_feats = condition_features["FBP"]
    results = []
    csv_rows = []

    # ---- binary classifier: FBP vs each ASIR level --------------------------
    for asir_label in [l for l in ASIR_ORDER if l != "FBP" and l in condition_features]:
        asir_feats = condition_features[asir_label]
        X = np.vstack([fbp_feats, asir_feats])
        y = np.array([0] * len(fbp_feats) + [1] * len(asir_feats))

        rf = RandomForestClassifier(n_estimators=RF_N_TREES,
                                    random_state=RNG_SEED, n_jobs=-1)
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RNG_SEED)

        # probabilities via CV
        y_prob = cross_val_predict(rf, X, y, cv=cv, method="predict_proba")[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        acc  = accuracy_score(y, y_pred)
        auc  = roc_auc_score(y, y_prob)
        fpr, tpr, _ = roc_curve(y, y_prob)
        cm   = confusion_matrix(y, y_pred)

        # fit on full data for feature importance
        rf.fit(X, y)
        imp = rf.feature_importances_

        comp = f"FBP vs {asir_label}"
        print(f"  {comp}: accuracy={acc:.3f}  AUC={auc:.3f}")

        results.append({
            "comparison":        comp,
            "asir_label":        asir_label,
            "accuracy":          acc,
            "auc":               auc,
            "fpr":               fpr,
            "tpr":               tpr,
            "confusion_matrix":  cm,
            "feature_importance": imp,
        })
        csv_rows.append({
            "comparison": comp,
            "n_fbp":      len(fbp_feats),
            "n_asir":     len(asir_feats),
            "accuracy":   round(acc, 4),
            "auc":        round(auc, 4),
            **{f"importance_{n}": round(float(v), 4)
               for n, v in zip(FEATURE_NAMES, imp)},
        })

    # ---- multi-class: all conditions simultaneously -------------------------
    all_labels = [l for l in ASIR_ORDER if l in condition_features]
    if len(all_labels) > 2:
        X_all = np.vstack([condition_features[l] for l in all_labels])
        y_all = np.concatenate([np.full(len(condition_features[l]), i)
                                 for i, l in enumerate(all_labels)])
        rf_mc = RandomForestClassifier(n_estimators=RF_N_TREES,
                                       random_state=RNG_SEED, n_jobs=-1)
        cv    = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RNG_SEED)
        y_pred_mc = cross_val_predict(rf_mc, X_all, y_all, cv=cv)
        acc_mc = accuracy_score(y_all, y_pred_mc)
        print(f"\n  Multi-class ({' / '.join(all_labels)}): accuracy={acc_mc:.3f}")
        csv_rows.append({
            "comparison": "Multi-class all conditions",
            "n_fbp": len(condition_features.get("FBP", [])),
            "n_asir": len(X_all) - len(condition_features.get("FBP", [])),
            "accuracy": round(acc_mc, 4),
            "auc": None,
        })

    # ---- save CSV -----------------------------------------------------------
    pd.DataFrame(csv_rows).to_csv(
        os.path.join(args.out, "domain_classifier.csv"), index=False)

    # ---- figures ------------------------------------------------------------
    print("\n--- Generating figures ---")
    plot_accuracy(results, os.path.join(args.out, "accuracy_by_asir.png"))
    plot_roc(results,      os.path.join(args.out, "roc_curves.png"))
    plot_feature_importance(results,
                            os.path.join(args.out, "feature_importance.png"))
    plot_confusion(results, os.path.join(args.out, "confusion_matrices.png"))

    # ---- print summary ------------------------------------------------------
    print("\n--- Summary ---")
    print("If a 4-feature RF distinguishes FBP from ASIR at high accuracy,")
    print("any AI trained on FBP images will encounter an equivalent distribution")
    print("shift on ASIR images -- with no signal from the ConvolutionKernel tag.\n")
    for r in results:
        pct_wrong = (1 - r["accuracy"]) * 100
        print(f"  {r['comparison']}: AUC={r['auc']:.3f}, "
              f"acc={r['accuracy']:.3f} "
              f"({pct_wrong:.1f}% of patches misclassified)")

    print("\nDone.")


if __name__ == "__main__":
    main()
