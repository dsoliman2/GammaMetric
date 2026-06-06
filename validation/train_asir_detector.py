"""Train the binary FBP-vs-iterative-reconstruction pixel classifier.

Labels FBP as 0, ALL ASIR levels (30/50/70%) as 1.  Saves to
models/asir_detector.joblib in the format expected by core/asir_detector.py.

Usage:
    python validation/train_asir_detector.py \
        --root G:/GammaMetric/manifest-1619103006849/QIBA-CT-Liver-Phantom \
        [--out  models/asir_detector.joblib]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score

# Reuse patch extraction from qiba_domain_classifier
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from validation.qiba_domain_classifier import (
    build_uid_map, load_central_slices, extract_patches,
    ASIR_SERIES, N_PATCHES, N_SLICES,
)

DEFAULT_OUT = os.path.join(os.path.dirname(__file__), '..', 'models', 'asir_detector.joblib')

# FBP = negative class (0); all ASIR levels = positive class (1)
FBP_KEYS  = ["FBP"]
ASIR_KEYS = ["ASIR 30%", "ASIR 50%", "ASIR 70%"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True,
                    help='QIBA-CT-Liver-Phantom DICOM root directory')
    ap.add_argument('--out', default=DEFAULT_OUT,
                    help='Output joblib path')
    ap.add_argument('--n-patches', type=int, default=N_PATCHES,
                    help='Patches per slice (default %(default)s)')
    args = ap.parse_args()

    print('Building UID -> directory map (may take a minute)...')
    uid_map = build_uid_map(args.root)

    X_all, y_all = [], []

    for label, keys in [(0, FBP_KEYS), (1, ASIR_KEYS)]:
        label_name = 'FBP' if label == 0 else 'ASIR (all levels)'
        for key in keys:
            uid = ASIR_SERIES[key]
            if uid not in uid_map:
                print(f'  WARNING: series not found for {key} ({uid}) — skipping')
                continue
            dicom_dir = uid_map[uid]
            print(f'  Loading {key}  ({uid[:40]}…)')
            slices = load_central_slices(dicom_dir, N_SLICES)
            feats  = extract_patches(slices, args.n_patches)
            X_all.append(feats)
            y_all.append(np.full(len(feats), label, dtype=int))
            print(f'    -> {len(feats)} patches  (label={label_name})')

    X = np.vstack(X_all)
    y = np.concatenate(y_all)
    print(f'\nTotal: {X.shape[0]} patches  (FBP={int((y==0).sum())}  ASIR={int((y==1).sum())})')

    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=10,
        n_jobs=-1,
        random_state=42,
        class_weight='balanced',
    )

    print('\nRunning 5-fold CV …')
    cv_aucs = cross_val_score(rf, Xs, y, cv=StratifiedKFold(5, shuffle=True, random_state=42),
                               scoring='roc_auc', n_jobs=-1)
    print(f'  CV AUC: {cv_aucs.mean():.4f} ± {cv_aucs.std():.4f}')

    print('Fitting final model on full dataset …')
    rf.fit(Xs, y)
    train_auc = roc_auc_score(y, rf.predict_proba(Xs)[:, 1])
    print(f'  Train AUC: {train_auc:.4f}')

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    joblib.dump({'model': rf, 'scaler': scaler, 'cv_auc_mean': float(cv_aucs.mean())}, out_path)
    print(f'\nSaved -> {out_path}')
    print(f'CV AUC: {cv_aucs.mean():.4f}  (use this number in docs/website)')


if __name__ == '__main__':
    main()
