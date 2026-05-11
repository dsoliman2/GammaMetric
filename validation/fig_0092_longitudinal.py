"""
Figure: LIDC-IDRI-0092, baseline vs 3mm slice thickness.

Demonstrates that a GREEN acquisition flag does not protect against
diameter measurement errors caused by protocol variation.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from validation.nodule_longitudinal import compare_studies

OUT_PATH = r'C:\Users\Dan\Desktop\gammametric_output\figures\fig_0092_longitudinal.png'
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

# ── Data ─────────────────────────────────────────────────────────────────────

result = compare_studies('LIDC-IDRI-0092', 'baseline', 'thick_3mm')

# Matched pairs (sorted by baseline diameter descending for readability)
pairs = sorted(result.matched, key=lambda p: p.study_a.diameter_mm, reverse=True)

# Append the disappeared nodule as a synthetic entry
disappeared = result.only_in_a  # detected baseline, missed thick_3mm
disappeared_diam = [d.diameter_mm for d in disappeared]

n_matched = len(pairs)
n = n_matched + len(disappeared)

labels     = [f'Nodule {p.nodule_id}' for p in pairs] + [f'Nodule {n_matched + i + 1}' for i in range(len(disappeared))]
diam_base  = [p.study_a.diameter_mm for p in pairs] + disappeared_diam
diam_thick = [p.study_b.diameter_mm for p in pairs]   # no entry for disappeared

deg_a = result.acq_a['degradation_pp']
deg_b = result.acq_b['degradation_pp']
cls_a = result.acq_a['classification']   # GREEN
cls_b = result.acq_b['classification']   # GREEN

# ── Style ────────────────────────────────────────────────────────────────────

COLOR_BASE  = '#4d4d4d'   # grey — baseline
COLOR_THICK = '#2c7bb6'   # blue — 3mm thick
COLOR_GREEN = '#2ecc71'
COLOR_WARN  = '#e74c3c'
FLEISCHNER_6  = 6.0   # mm — below: no routine follow-up
FLEISCHNER_8  = 8.0   # mm — above: aggressive follow-up

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size':   11,
    'axes.spines.top':   False,
    'axes.spines.right': False,
})

fig = plt.figure(figsize=(12, 5.5))
fig.patch.set_facecolor('white')

# 1/3 for acquisition panel, 2/3 for bar chart
gs = fig.add_gridspec(1, 3, wspace=0.35, left=0.07, right=0.97, top=0.88, bottom=0.14)
ax_acq = fig.add_subplot(gs[0, 0])
ax_bar = fig.add_subplot(gs[0, 1:])

# ── Panel A: Acquisition reliability ─────────────────────────────────────────

ax_acq.set_xlim(0, 1)
ax_acq.set_ylim(0, 1)
ax_acq.axis('off')

def acq_block(ax, x, y, label, deg, cls):
    color = COLOR_GREEN if cls == 'GREEN' else COLOR_WARN
    # outer box
    rect = mpatches.FancyBboxPatch(
        (x - 0.28, y - 0.13), 0.56, 0.26,
        boxstyle='round,pad=0.02',
        facecolor=color, edgecolor='white', alpha=0.15, linewidth=0,
        transform=ax.transAxes,
    )
    ax.add_patch(rect)
    ax.text(x, y + 0.06, label, ha='center', va='center',
            fontsize=10, color='#333333', transform=ax.transAxes)
    ax.text(x, y - 0.04, cls, ha='center', va='center',
            fontsize=15, fontweight='bold', color=color, transform=ax.transAxes)
    ax.text(x, y - 0.11, f'{deg:.1f} pp degradation', ha='center', va='center',
            fontsize=8.5, color='#666666', transform=ax.transAxes)

ax_acq.text(0.5, 0.93, 'Acquisition Check', ha='center', va='top',
            fontsize=12, fontweight='bold', color='#222222', transform=ax_acq.transAxes)

acq_block(ax_acq, 0.28, 0.67, 'Baseline\n1.25 mm slices', deg_a, cls_a)
acq_block(ax_acq, 0.72, 0.67, '3 mm Slices', deg_b, cls_b)

# Arrow between them
ax_acq.annotate('', xy=(0.54, 0.67), xytext=(0.46, 0.67),
                arrowprops=dict(arrowstyle='->', color='#aaaaaa', lw=1.5),
                xycoords='axes fraction', textcoords='axes fraction')

# Warning box
warn_rect = mpatches.FancyBboxPatch(
    (0.04, 0.05), 0.92, 0.38,
    boxstyle='round,pad=0.02',
    facecolor='#fef9e7', edgecolor='#f39c12', linewidth=1.5,
    transform=ax_acq.transAxes,
)
ax_acq.add_patch(warn_rect)
ax_acq.text(0.5, 0.39, 'Both GREEN', ha='center', va='top',
            fontsize=10.5, fontweight='bold', color='#e67e22', transform=ax_acq.transAxes)
ax_acq.text(0.5, 0.30, 'Standard QC would approve\nthis comparison as reliable.',
            ha='center', va='top', fontsize=9, color='#555555',
            transform=ax_acq.transAxes, linespacing=1.5)
ax_acq.text(0.5, 0.14, 'But diameter measurements\nshow 40-65% apparent growth.',
            ha='center', va='top', fontsize=9, color=COLOR_WARN,
            fontweight='bold', transform=ax_acq.transAxes, linespacing=1.5)

ax_acq.text(0.5, -0.02, '(A)', ha='center', va='bottom',
            fontsize=10, color='#888888', transform=ax_acq.transAxes)

# ── Panel B: Diameter comparison bar chart ───────────────────────────────────

x      = np.arange(n)
width  = 0.32

# Matched nodule bars
bars_b = ax_bar.bar(x[:n_matched] - width/2, diam_base[:n_matched],
                    width, color=COLOR_BASE,  label='Baseline (1.25 mm)',   zorder=3)
bars_t = ax_bar.bar(x[:n_matched] + width/2, diam_thick,
                    width, color=COLOR_THICK, label='3 mm Slice Thickness', zorder=3)

# Disappeared nodule — baseline bar only
for i in range(len(disappeared)):
    idx = n_matched + i
    ax_bar.bar(x[idx] - width/2, diam_base[idx], width,
               color=COLOR_BASE, zorder=3)
    # Hatched placeholder where thick_3mm bar would be
    ax_bar.bar(x[idx] + width/2, diam_base[idx] * 0.85, width,
               color='none', edgecolor='#aaaaaa', linewidth=1.2,
               linestyle='--', hatch='////', zorder=3, alpha=0.6)
    # "Not detected" X marker
    ax_bar.text(x[idx] + width/2, diam_base[idx] * 0.85 / 2,
                'Not\ndetected', ha='center', va='center',
                fontsize=8, color='#888888', style='italic')

# Value labels on baseline bars
for bar, val in zip(bars_b, diam_base[:n_matched]):
    ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val:.1f}', ha='center', va='bottom', fontsize=9, color=COLOR_BASE)
# Disappeared nodule baseline label
for i in range(len(disappeared)):
    idx = n_matched + i
    ax_bar.text(x[idx] - width/2, diam_base[idx] + 0.3,
                f'{diam_base[idx]:.1f}', ha='center', va='bottom',
                fontsize=9, color=COLOR_BASE)

for bar, val in zip(bars_t, diam_thick):
    ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val:.1f}', ha='center', va='bottom', fontsize=9,
                color=COLOR_THICK, fontweight='bold')

# Delta annotations (arrows + percentage) for matched pairs only
for i, p in enumerate(pairs):
    x_base  = x[i] - width/2
    x_thick = x[i] + width/2
    y_bot   = p.study_a.diameter_mm
    y_top   = p.study_b.diameter_mm
    x_mid   = (x_base + x_thick) / 2

    ax_bar.annotate('', xy=(x_thick, y_top + 0.3), xytext=(x_base, y_bot + 0.3),
                    arrowprops=dict(arrowstyle='->', color=COLOR_WARN,
                                   lw=1.4, connectionstyle='arc3,rad=-0.25'))
    pct = p.diameter_delta_pct
    ax_bar.text(x_mid + 0.18, (y_bot + y_top)/2 + 1.5,
                f'+{pct:.0f}%', ha='center', va='bottom',
                fontsize=9, color=COLOR_WARN, fontweight='bold')

# Disappeared nodule annotation
for i in range(len(disappeared)):
    idx = n_matched + i
    ax_bar.annotate('', xy=(x[idx] + width/2, diam_base[idx] * 0.85 + 0.3),
                    xytext=(x[idx] - width/2, diam_base[idx] + 0.3),
                    arrowprops=dict(arrowstyle='->', color=COLOR_WARN,
                                   lw=1.4, connectionstyle='arc3,rad=-0.25'))
    ax_bar.text(x[idx] + 0.22, diam_base[idx] + 0.8,
                'Not matched\non follow-up', ha='center', va='bottom',
                fontsize=8.5, color=COLOR_WARN, fontweight='bold')

# Fleischner threshold lines
xmin = -0.5
xmax = n - 0.5
ax_bar.axhline(FLEISCHNER_6, xmin=0, xmax=1, color='#7f8c8d', lw=1.2,
               ls='--', zorder=2, alpha=0.8)
ax_bar.axhline(FLEISCHNER_8, xmin=0, xmax=1, color='#7f8c8d', lw=1.2,
               ls=':', zorder=2, alpha=0.8)

ax_bar.text(n - 0.08, FLEISCHNER_6 + 0.2, '6 mm  (Fleischner threshold)',
            ha='right', va='bottom', fontsize=8, color='#7f8c8d', style='italic')
ax_bar.text(n - 0.08, FLEISCHNER_8 + 0.2, '8 mm  (high-risk threshold)',
            ha='right', va='bottom', fontsize=8, color='#7f8c8d', style='italic')

# Axes
ax_bar.set_xticks(x)
ax_bar.set_xticklabels(labels, fontsize=11)
ax_bar.set_ylabel('Measured Diameter (mm)', fontsize=11)
ax_bar.set_ylim(0, max(diam_thick) * 1.22)
ax_bar.set_xlim(-0.55, n - 0.38)
ax_bar.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(1))
ax_bar.grid(axis='y', alpha=0.25, zorder=0)
ax_bar.legend(loc='upper left', fontsize=9.5, framealpha=0.9)

ax_bar.text(0.5, -0.02, '(B)', ha='center', va='bottom',
            fontsize=10, color='#888888', transform=ax_bar.transAxes)

# ── Title ─────────────────────────────────────────────────────────────────────

fig.text(0.5, 0.97,
         'Apparent Nodule Growth Caused by Protocol Change — LIDC-IDRI-0092',
         ha='center', va='top', fontsize=13, fontweight='bold', color='#1a1a1a')
fig.text(0.5, 0.925,
         'Both acquisition conditions flagged GREEN by the sensitivity engine, '
         'yet diameter measurements differ by 40-65% and one nodule disappears entirely.',
         ha='center', va='top', fontsize=10, color='#555555')

# ── Save ─────────────────────────────────────────────────────────────────────

plt.savefig(OUT_PATH, dpi=180, bbox_inches='tight', facecolor='white')
print(f'Saved: {OUT_PATH}')
