"""
GammaMetric — Detection Count + Confidence Heatmap
Reads corrected per-case result JSONs and plots:
  Left:  Detection Count by Condition (stacked bar)
  Right: Detection Confidence by Nodule (heatmap grid)
"""

import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

BASE         = r'C:\Users\Dan\Desktop\gammametric_output'
RESULTS_BASE = r'C:\Users\Dan\Desktop\gammametric_output\nodule_results_each'
CASE_ID      = 'LIDC-IDRI-0009'
THRESHOLD    = 0.5

CONDITIONS = ['baseline', 'dose_25pct', 'dose_50pct', 'thick_3mm', 'thick_5mm']
LABELS = {
    'baseline':   'Baseline\n(Standard)',
    'dose_25pct': '25% Dose\nReduction',
    'dose_50pct': '50% Dose\nReduction',
    'thick_3mm':  '3mm Slice\nThickness',
    'thick_5mm':  '5mm Slice\nThickness',
}

def load_result(case_id, condition):
    # new per-case path
    path = os.path.join(RESULTS_BASE, f'{case_id}_{condition}', 'result_luna16_fold0.json')
    if not os.path.exists(path):
        # 0009 legacy fallback
        path = os.path.join(RESULTS_BASE, condition, 'result_luna16_fold0.json')
    with open(path) as f:
        data = json.load(f)
    # handle both list and dict formats
    if isinstance(data, list):
        data = data[0]
    elif 'value' in data:
        data = data['value'][0]
    return data['box'], data['label_scores']

# Load all results
results = {}
for cond in CONDITIONS:
    try:
        boxes, scores = load_result(CASE_ID, cond)
        results[cond] = sorted(scores, reverse=True)
        print(f'{LABELS[cond].replace(chr(10)," ")}: {[f"{s:.2f}" for s in results[cond]]}')
    except FileNotFoundError:
        print(f'Missing: {cond}')
        results[cond] = None

# Determine max nodule count across conditions
max_nodules = max(len(s) for s in results.values() if s is not None)
n_conds = len(CONDITIONS)

# Build score matrix (rows=conditions, cols=nodules), pad with NaN
score_matrix = np.full((n_conds, max_nodules), np.nan)
for i, cond in enumerate(CONDITIONS):
    if results[cond] is not None:
        for j, s in enumerate(results[cond]):
            score_matrix[i, j] = s

# ─── FIGURE ───────────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(14, 6))
fig.patch.set_facecolor('#0f1117')

# Title
fig.text(0.5, 0.97,
         'GammaMetric  |  AI Robustness Validation  |  Lung Nodule Detection',
         ha='center', va='top', color='white', fontsize=11, fontweight='bold')
fig.text(0.5, 0.91,
         f'LIDC-IDRI Case  ·  TotalSegmentator RetinaNet  ·  MONAI',
         ha='center', va='top', color='#888888', fontsize=8)

gs = fig.add_gridspec(1, 2, left=0.07, right=0.97,
                      top=0.86, bottom=0.12, wspace=0.35)

ax_bar  = fig.add_subplot(gs[0, 0])
ax_heat = fig.add_subplot(gs[0, 1])

for ax in [ax_bar, ax_heat]:
    ax.set_facecolor('#1a1a2e')

# ─── LEFT: stacked bar ────────────────────────────────────────────────────────

x = np.arange(n_conds)
high_counts = []
low_counts  = []
for cond in CONDITIONS:
    if results[cond] is None:
        high_counts.append(0)
        low_counts.append(0)
    else:
        high_counts.append(sum(s >= THRESHOLD for s in results[cond]))
        low_counts.append(sum(s < THRESHOLD for s in results[cond]))

bars_high = ax_bar.bar(x, high_counts, color='#00CC66', label='High confidence (≥0.5)', zorder=3)
bars_low  = ax_bar.bar(x, low_counts, bottom=high_counts, color='#CC3333',
                        label='Low confidence (<0.5)', zorder=3)

# Value labels
for i, (h, l) in enumerate(zip(high_counts, low_counts)):
    total = h + l
    if h > 0:
        ax_bar.text(i, h / 2, str(h), ha='center', va='center',
                    color='white', fontsize=11, fontweight='bold', zorder=4)
    if l > 0:
        ax_bar.text(i, h + l / 2, str(l), ha='center', va='center',
                    color='white', fontsize=11, fontweight='bold', zorder=4)

ax_bar.set_xticks(x)
ax_bar.set_xticklabels([LABELS[c] for c in CONDITIONS],
                        color='white', fontsize=8)
ax_bar.set_ylabel('Nodule Candidates', color='white', fontsize=9)
ax_bar.set_title('Detection Count by Condition', color='white',
                  fontsize=10, fontweight='bold', pad=8)
ax_bar.tick_params(colors='white')
ax_bar.spines['bottom'].set_color('#444')
ax_bar.spines['left'].set_color('#444')
ax_bar.spines['top'].set_visible(False)
ax_bar.spines['right'].set_visible(False)
ax_bar.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
ax_bar.set_ylim(0, max(h+l for h, l in zip(high_counts, low_counts)) + 1.5)
ax_bar.legend(fontsize=7, facecolor='#1a1a2e', labelcolor='white',
              loc='upper right', framealpha=0.8)
ax_bar.grid(axis='y', color='#333', linestyle='--', alpha=0.5, zorder=0)

# ─── RIGHT: confidence heatmap ────────────────────────────────────────────────

# Custom colormap: red → yellow → green
cmap = mcolors.LinearSegmentedColormap.from_list(
    'rg', ['#CC2222', '#FFAA00', '#00AA44'], N=256)
cmap.set_bad(color='#222233')  # NaN = dark

im = ax_heat.imshow(score_matrix, cmap=cmap, vmin=0, vmax=1,
                     aspect='auto', interpolation='nearest')

# Score annotations
for i in range(n_conds):
    for j in range(max_nodules):
        val = score_matrix[i, j]
        if not np.isnan(val):
            color = 'white' if val < 0.7 else '#0a0a0a'
            ax_heat.text(j, i, f'{val:.2f}', ha='center', va='center',
                          fontsize=9, fontweight='bold', color=color)
            # box border if above threshold
            if val >= THRESHOLD:
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                      linewidth=2, edgecolor='#00FF88',
                                      facecolor='none')
                ax_heat.add_patch(rect)

ax_heat.set_xticks(range(max_nodules))
ax_heat.set_xticklabels([f'Nodule {j+1}' for j in range(max_nodules)],
                          color='white', fontsize=9)
ax_heat.set_yticks(range(n_conds))
ax_heat.set_yticklabels([LABELS[c] for c in CONDITIONS],
                          color='white', fontsize=8)
ax_heat.set_title('Detection Confidence by Nodule', color='white',
                   fontsize=10, fontweight='bold', pad=8)
ax_heat.tick_params(colors='white', length=0)
for spine in ax_heat.spines.values():
    spine.set_color('#444')

# Colorbar
cbar = fig.colorbar(im, ax=ax_heat, fraction=0.03, pad=0.02)
cbar.ax.yaxis.set_tick_params(color='white')
cbar.outline.set_edgecolor('#444')
plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white', fontsize=7)
cbar.set_label('Confidence Score', color='white', fontsize=8)

# ─── SAVE ─────────────────────────────────────────────────────────────────────

out = os.path.join(BASE, 'nodule_heatmap_corrected.png')
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f'Saved: {out}')
