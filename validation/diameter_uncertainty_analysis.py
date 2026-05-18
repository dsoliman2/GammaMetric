"""
Empirical diameter uncertainty analysis.

For each LIDC-IDRI case, matches nodules between baseline and each degraded
condition, then characterizes how apparent diameter shifts under acquisition
parameter changes.

Outputs:
  - diameter_shifts.csv  — per-nodule shifts for all matched pairs
  - diameter_model.json  — mean shift + uncertainty bounds per condition
    (used to populate sensitivity_engine.py diameter uncertainty table)

Usage:
  python diameter_uncertainty_analysis.py
"""

import os
import json
import glob
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

RESULTS_BASE = r'C:\Users\Dan\Desktop\gammametric_output\nodule_results_each'
OUTPUT_DIR   = r'G:\GammaMetric\diameter_analysis'
CONF_THRESH  = 0.25   # lower threshold to capture near-threshold nodules
MAX_MATCH_MM = 20.0   # max centroid distance for matching
MIN_DIAM_MM  = 3.0    # ignore tiny detections (likely FPs)

CONDITIONS = {
    'dose_25pct':   {'ctdivol_mgy': 7.5,  'slice_mm': None, 'kernel': None},
    'dose_50pct':   {'ctdivol_mgy': 5.0,  'slice_mm': None, 'kernel': None},
    'thick_3mm':    {'ctdivol_mgy': None, 'slice_mm': 3.0,  'kernel': None},
    'thick_5mm':    {'ctdivol_mgy': None, 'slice_mm': 5.0,  'kernel': None},
    'soft_kernel':  {'ctdivol_mgy': None, 'slice_mm': None, 'kernel': 'SOFT'},
    'sharp_kernel': {'ctdivol_mgy': None, 'slice_mm': None, 'kernel': 'SHARP'},
}

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_detections(result_dir, conf_thresh=CONF_THRESH):
    path = os.path.join(result_dir, 'result_luna16_fold0.json')
    if not os.path.exists(path):
        return []
    with open(path) as f:
        data = json.load(f)
    if not data:
        return []
    dets = []
    boxes  = data[0].get('box', [])
    scores = data[0].get('label_scores', [])
    for box, score in zip(boxes, scores):
        if score < conf_thresh:
            continue
        cx, cy, cz, w, h, d = box
        diam = max(w, h, d)
        if diam < MIN_DIAM_MM:
            continue
        dets.append({'cx': cx, 'cy': cy, 'cz': cz,
                     'w': w, 'h': h, 'd': d,
                     'diameter': diam, 'score': score})
    return dets


def match_detections(dets_a, dets_b, max_dist=MAX_MATCH_MM):
    """Hungarian matching on centroid distance."""
    if not dets_a or not dets_b:
        return []
    centers_a = np.array([[d['cx'], d['cy'], d['cz']] for d in dets_a])
    centers_b = np.array([[d['cx'], d['cy'], d['cz']] for d in dets_b])
    dist = np.linalg.norm(centers_a[:, None] - centers_b[None, :], axis=2)
    row_idx, col_idx = linear_sum_assignment(dist)
    pairs = []
    for r, c in zip(row_idx, col_idx):
        if dist[r, c] <= max_dist:
            pairs.append((dets_a[r], dets_b[c], float(dist[r, c])))
    return pairs


def analyze():
    baseline_dirs = sorted(glob.glob(os.path.join(RESULTS_BASE, '*_baseline')))
    rows = []

    for baseline_dir in baseline_dirs:
        case_id = os.path.basename(baseline_dir).replace('_baseline', '')
        dets_base = load_detections(baseline_dir)
        if not dets_base:
            continue

        for cond_suffix, params in CONDITIONS.items():
            cond_dir = os.path.join(RESULTS_BASE, f'{case_id}_{cond_suffix}')
            if not os.path.exists(cond_dir):
                continue
            dets_cond = load_detections(cond_dir)
            pairs = match_detections(dets_base, dets_cond)

            for det_base, det_cond, centroid_dist in pairs:
                d_base = det_base['diameter']
                d_cond = det_cond['diameter']
                rows.append({
                    'case_id':       case_id,
                    'condition':     cond_suffix,
                    'diam_baseline': round(d_base, 2),
                    'diam_cond':     round(d_cond, 2),
                    'delta_diam':    round(d_cond - d_base, 2),
                    'centroid_dist': round(centroid_dist, 2),
                    'score_base':    round(det_base['score'], 3),
                    'score_cond':    round(det_cond['score'], 3),
                })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUTPUT_DIR, 'diameter_shifts.csv')
    df.to_csv(csv_path, index=False)
    print(f'Matched pairs: {len(df)} across {df["case_id"].nunique()} cases')

    # Summary statistics per condition
    print('\nDiameter shift statistics per condition:')
    print(f'{"Condition":<14} {"N":>5} {"Mean Δ":>8} {"Std Δ":>8} {"p5":>8} {"p95":>8} {"abs_mean":>10}')
    model = {}
    for cond in CONDITIONS:
        sub = df[df['condition'] == cond]['delta_diam']
        if len(sub) == 0:
            continue
        mean  = sub.mean()
        std   = sub.std()
        p5    = sub.quantile(0.05)
        p95   = sub.quantile(0.95)
        absmn = sub.abs().mean()
        print(f'{cond:<14} {len(sub):>5} {mean:>+8.2f} {std:>8.2f} {p5:>+8.2f} {p95:>+8.2f} {absmn:>10.2f}')
        model[cond] = {
            'mean_shift_mm':       round(float(mean), 2),
            'std_mm':              round(float(std), 2),
            'p5_mm':               round(float(p5), 2),
            'p95_mm':              round(float(p95), 2),
            'uncertainty_95ci_mm': round(float(p95 - p5), 2),
            'n_pairs':             int(len(sub)),
        }

    # Size-stratified analysis
    print('\nDiameter shift by nodule size (baseline diameter):')
    bins = [(3, 6), (6, 10), (10, 20), (20, 50)]
    for cond in CONDITIONS:
        sub = df[df['condition'] == cond]
        if len(sub) == 0:
            continue
        print(f'\n  {cond}:')
        for lo, hi in bins:
            s = sub[(sub['diam_baseline'] >= lo) & (sub['diam_baseline'] < hi)]['delta_diam']
            if len(s) < 5:
                continue
            print(f'    {lo}-{hi}mm  n={len(s):3d}  Δ={s.mean():+.2f}±{s.std():.2f}mm  95CI=[{s.quantile(0.05):+.2f},{s.quantile(0.95):+.2f}]')
        if cond in model:
            model[cond]['size_strata'] = {}
            for lo, hi in bins:
                s = sub[(sub['diam_baseline'] >= lo) & (sub['diam_baseline'] < hi)]['delta_diam']
                if len(s) >= 5:
                    model[cond]['size_strata'][f'{lo}-{hi}mm'] = {
                        'mean_shift_mm': round(float(s.mean()), 2),
                        'std_mm':        round(float(s.std()), 2),
                        'uncertainty_95ci_mm': round(float(s.quantile(0.95) - s.quantile(0.05)), 2),
                        'n': int(len(s)),
                    }

    model_path = os.path.join(OUTPUT_DIR, 'diameter_model.json')
    with open(model_path, 'w') as f:
        json.dump(model, f, indent=2)
    print(f'\nModel saved: {model_path}')
    return df, model


if __name__ == '__main__':
    df, model = analyze()
