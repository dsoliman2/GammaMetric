"""
Validates MONAI RetinaNet bounding-box diameter estimates against
LIDC-IDRI radiologist annotations (pylidc consensus).

For each preprint case / baseline condition:
  1. Load DICOM image geometry (origin, spacing) to convert pylidc
     pixel centroids -> world mm.
  2. Pull pylidc consensus nodule diameters (mean across readers).
  3. Load MONAI baseline detections; match to LIDC nodules by centroid
     proximity (<= MATCH_DIST_MM).
  4. Record paired (lidc_diam, bbox_diam) for all matched nodules.

Outputs:
  - Console table of all matched pairs
  - Scatter plot + Bland-Altman figure
  - Summary stats: bias, LoA, correlation
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pylidc as pl
import SimpleITK as sitk

DICOM_BASE   = r'C:\Users\Dan\Desktop\gammametric_output\Lung DICOM\manifest-1600709154662\LIDC-IDRI'
RESULTS_BASE = r'C:\Users\Dan\Desktop\gammametric_output\nodule_results_each'
OUT_PATH     = r'C:\Users\Dan\Desktop\gammametric_output\figures\diameter_validation.png'

import glob as _glob
PREPRINT_CASES = sorted(set(
    os.path.basename(d).replace('_baseline', '')
    for d in _glob.glob(os.path.join(RESULTS_BASE, '*_baseline'))
))
MIN_READERS    = 2
CONF_THRESHOLD = 0.5
MATCH_DIST_MM  = 15.0
MAX_LIDC_DIAM  = 15.0   # restrict to Fleischner-relevant nodules


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_largest_series(case_id: str) -> str | None:
    case_dir = os.path.join(DICOM_BASE, case_id)
    best_dir, best_count = None, 0
    for dirpath, _, files in os.walk(case_dir):
        dcms = [f for f in files if f.endswith('.dcm')]
        if len(dcms) > best_count:
            best_count = len(dcms)
            best_dir = dirpath
    return best_dir


def load_dicom_image(case_id: str):
    dicom_dir = find_largest_series(case_id)
    if not dicom_dir:
        return None
    reader = sitk.ImageSeriesReader()
    files = reader.GetGDCMSeriesFileNames(dicom_dir)
    if not files:
        return None
    reader.SetFileNames(files)
    return reader.Execute()


def lidc_consensus_nodules(case_id: str, img, min_readers: int = MIN_READERS) -> list[dict]:
    """
    Returns list of dicts with world-mm centroid and mean radiologist diameter.
    pylidc centroid is (row, col, slice_idx); ITK/SimpleITK index is (col, row, slice).
    """
    scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == case_id).first()
    if scan is None:
        return []
    try:
        clusters = scan.cluster_annotations(tol=5)
    except Exception:
        clusters = scan.cluster_annotations()

    nodules = []
    for cluster in clusters:
        if len(cluster) < min_readers:
            continue
        if np.mean([ann.diameter for ann in cluster]) > MAX_LIDC_DIAM:
            continue
        diams = [ann.diameter for ann in cluster]
        centroids = np.array([ann.centroid for ann in cluster])  # (N, 3): row, col, slice
        mean_c = centroids.mean(axis=0)
        # swap row/col for ITK index convention
        try:
            world = img.TransformIndexToPhysicalPoint(
                (int(round(mean_c[1])), int(round(mean_c[0])), int(round(mean_c[2])))
            )
        except Exception:
            continue
        nodules.append({
            'world_mm':    world,
            'lidc_diam':   float(np.mean(diams)),
            'reader_count': len(cluster),
        })
    return nodules


def load_baseline_detections(case_id: str) -> list[dict]:
    path = os.path.join(RESULTS_BASE, f'{case_id}_baseline', 'result_luna16_fold0.json')
    if not os.path.exists(path):
        return []
    with open(path) as f:
        raw = json.load(f)
    data = raw['value'][0] if isinstance(raw, dict) and 'value' in raw else raw[0]
    dets = []
    for box, score in zip(data.get('box', []), data.get('label_scores', [])):
        if score < CONF_THRESHOLD:
            continue
        dets.append({
            'center': (float(box[0]), float(box[1]), float(box[2])),
            'bbox_diam': float(max(box[3], box[4], box[5])),
            'confidence': float(score),
        })
    return dets


def match_to_lidc(nodules: list[dict], dets: list[dict]) -> list[dict]:
    used = [False] * len(dets)
    pairs = []
    for nod in nodules:
        nx, ny, nz = nod['world_mm']
        best_idx, best_dist = None, float('inf')
        for j, det in enumerate(dets):
            if used[j]:
                continue
            dx, dy, dz = det['center']
            dist = float(np.sqrt((nx-dx)**2 + (ny-dy)**2 + (nz-dz)**2))
            if dist < MATCH_DIST_MM and dist < best_dist:
                best_dist = dist
                best_idx = j
        if best_idx is not None:
            used[best_idx] = True
            pairs.append({
                'lidc_diam':    nod['lidc_diam'],
                'bbox_diam':    dets[best_idx]['bbox_diam'],
                'confidence':   dets[best_idx]['confidence'],
                'dist_mm':      best_dist,
                'reader_count': nod['reader_count'],
            })
    return pairs


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

all_pairs: list[dict] = []

print(f'\n{"Case":<22} {"LIDC d":>8} {"BBox d":>8} {"Bias":>8} {"Dist":>7} {"Conf":>7} {"Rdrs"}')
print('-' * 72)

for case_id in PREPRINT_CASES:
    img = load_dicom_image(case_id)
    if img is None:
        print(f'{case_id:<22}  no DICOM')
        continue

    nodules = lidc_consensus_nodules(case_id, img)
    dets    = load_baseline_detections(case_id)

    if not nodules:
        print(f'{case_id:<22}  no pylidc nodules')
        continue
    if not dets:
        print(f'{case_id:<22}  no baseline detections')
        continue

    pairs = match_to_lidc(nodules, dets)
    for p in pairs:
        bias = p['bbox_diam'] - p['lidc_diam']
        print(f'{case_id:<22} {p["lidc_diam"]:>7.1f}mm {p["bbox_diam"]:>7.1f}mm '
              f'{bias:>+7.1f}mm {p["dist_mm"]:>6.1f}mm {p["confidence"]:>6.3f}  {p["reader_count"]}')
        all_pairs.append({**p, 'case_id': case_id})

print('-' * 72)

if not all_pairs:
    print('No matched pairs found.')
    sys.exit(1)

lidc_d = np.array([p['lidc_diam']  for p in all_pairs])
bbox_d = np.array([p['bbox_diam']  for p in all_pairs])
bias   = bbox_d - lidc_d
mean_d = (bbox_d + lidc_d) / 2

mean_bias = float(bias.mean())
sd_bias   = float(bias.std(ddof=1))
loa_lo    = mean_bias - 1.96 * sd_bias
loa_hi    = mean_bias + 1.96 * sd_bias
r         = float(np.corrcoef(lidc_d, bbox_d)[0, 1])

print(f'\nN matched pairs : {len(all_pairs)}')
print(f'Mean bias       : {mean_bias:+.2f} mm  (bbox - LIDC)')
print(f'SD of bias      : {sd_bias:.2f} mm')
print(f'95% LoA         : [{loa_lo:+.2f}, {loa_hi:+.2f}] mm')
print(f'Pearson r       : {r:.3f}')


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 11,
                     'axes.spines.top': False, 'axes.spines.right': False})

fig, (ax_sc, ax_ba) = plt.subplots(1, 2, figsize=(11, 5))
fig.patch.set_facecolor('white')
fig.suptitle('MONAI Bounding-Box Diameter vs. LIDC-IDRI Radiologist Annotation\n'
             f'Baseline condition, {len(all_pairs)} matched nodules ≤15 mm across {len(PREPRINT_CASES)} cases',
             fontsize=12, fontweight='bold', y=0.98)

COLOR = '#2c7bb6'

# ── Scatter ──────────────────────────────────────────────────────────────────
lim = (0, max(lidc_d.max(), bbox_d.max()) * 1.12)
ax_sc.plot(lim, lim, '--', color='#aaaaaa', lw=1.2, label='Identity')
ax_sc.scatter(lidc_d, bbox_d, color=COLOR, s=55, zorder=3, alpha=0.85)

# per-nodule labels
for p in all_pairs:
    ax_sc.annotate(f'{p["case_id"].split("-")[-1]}',
                   xy=(p['lidc_diam'], p['bbox_diam']),
                   xytext=(3, 3), textcoords='offset points',
                   fontsize=7, color='#555555')

ax_sc.set_xlim(lim)
ax_sc.set_ylim(lim)
ax_sc.set_xlabel('LIDC Annotated Diameter (mm)')
ax_sc.set_ylabel('MONAI BBox Diameter (mm)')
ax_sc.set_title('(A)  Scatter', loc='left', fontsize=11, fontweight='bold')
ax_sc.text(0.05, 0.93, f'r = {r:.3f}', transform=ax_sc.transAxes,
           fontsize=10, color='#333333')
ax_sc.legend(fontsize=9)

# ── Bland-Altman ─────────────────────────────────────────────────────────────
ax_ba.axhline(mean_bias, color=COLOR,      lw=1.8, label=f'Bias {mean_bias:+.2f} mm')
ax_ba.axhline(loa_hi,    color='#e74c3c',  lw=1.2, ls='--', label=f'+1.96 SD  {loa_hi:+.2f} mm')
ax_ba.axhline(loa_lo,    color='#e74c3c',  lw=1.2, ls='--', label=f'-1.96 SD  {loa_lo:+.2f} mm')
ax_ba.axhline(0,         color='#aaaaaa',  lw=0.9, ls=':')
ax_ba.scatter(mean_d, bias, color=COLOR, s=55, zorder=3, alpha=0.85)

ax_ba.set_xlabel('Mean of LIDC and BBox Diameter (mm)')
ax_ba.set_ylabel('BBox - LIDC Diameter (mm)')
ax_ba.set_title('(B)  Bland-Altman', loc='left', fontsize=11, fontweight='bold')
ax_ba.legend(fontsize=9, loc='lower right')

plt.tight_layout(rect=[0, 0, 1, 0.93])
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
plt.savefig(OUT_PATH, dpi=180, bbox_inches='tight', facecolor='white')
print(f'\nSaved: {OUT_PATH}')
