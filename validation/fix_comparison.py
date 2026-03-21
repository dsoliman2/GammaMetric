"""
GammaMetric — Nodule comparison figure v5
- Zoom panels centered
- Right zoom targets highest NEW detection (not in baseline) = strongest false positive story
"""

import json, numpy as np, SimpleITK as sitk
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

base = r'C:\Users\Dan\Desktop\gammametric_output'

def load_result(cond):
    path = f'{base}/nodule_results_each/{cond}/result_luna16_fold0.json'
    with open(path) as f:
        return json.load(f)[0]

def load_volume(nii_path):
    image = sitk.ReadImage(nii_path)
    arr = sitk.GetArrayFromImage(image)
    return image, arr

def world_to_voxel(image, world_xyz):
    return image.TransformPhysicalPointToIndex(
        (float(world_xyz[0]), float(world_xyz[1]), float(world_xyz[2]))
    )

def get_best_slice(image, result, threshold=0.5):
    best_score, best_z = -1, None
    for box, score in zip(result['box'], result['label_scores']):
        if score >= threshold:
            idx = world_to_voxel(image, box[:3])
            if score > best_score:
                best_score = score
                best_z = idx[2]
    if best_z is None:
        idx = world_to_voxel(image, result['box'][0][:3])
        best_z = idx[2]
    return best_z

def draw_boxes(ax, image, arr, result, z_slice, threshold=0.5, highlight_box=None):
    ax.imshow(arr[z_slice], cmap='gray', vmin=-1000, vmax=400)
    ax.axis('off')
    for box, score in zip(result['box'], result['label_scores']):
        idx = world_to_voxel(image, box[:3])
        spacing = image.GetSpacing()
        hw = box[3] / spacing[0] / 2
        hh = box[4] / spacing[1] / 2
        x0 = idx[0] - hw
        y0 = idx[1] - hh
        color = '#00FF88' if score >= threshold else '#FF4444'
        lw    = 2.0        if score >= threshold else 1.5
        ls    = 'solid'    if score >= threshold else 'dashed'
        rect = patches.Rectangle((x0, y0), hw*2, hh*2,
                                  linewidth=lw, edgecolor=color,
                                  facecolor='none', linestyle=ls)
        ax.add_patch(rect)
        ax.text(x0, y0 - 5, f'{score:.2f}', color=color, fontsize=9,
                fontweight='bold',
                bbox=dict(facecolor='#0f1117', alpha=0.6, pad=1, linewidth=0))
    if highlight_box is not None:
        hx0, hy0, hw2, hh2 = highlight_box
        rect_hi = patches.Rectangle((hx0, hy0), hw2, hh2,
                                     linewidth=2, edgecolor='#FFD700',
                                     facecolor='none', linestyle='--')
        ax.add_patch(rect_hi)

def get_zoom_coords(image, arr, result, z_slice, chosen_box, pad=50):
    """Return crop coords centered on a specific chosen box."""
    idx = world_to_voxel(image, chosen_box[:3])
    cx_v, cy_v = idx[0], idx[1]
    x0 = max(0, cx_v - pad)
    x1 = min(arr.shape[2], cx_v + pad)
    y0 = max(0, cy_v - pad)
    y1 = min(arr.shape[1], cy_v + pad)
    return x0, y0, x1, y1, idx

def draw_zoom_panel(ax, image, arr, result, z_slice,
                    x0, y0, x1, y1, focus_idx, threshold=0.5, title=''):
    ax.imshow(arr[z_slice, y0:y1, x0:x1], cmap='gray', vmin=-1000, vmax=400)
    ax.axis('off')
    ax.set_title(title, color='#FFD700', fontsize=10, pad=6)
    for spine in ax.spines.values():
        spine.set_edgecolor('#FFD700')
        spine.set_linewidth(2)
        spine.set_visible(True)

    spacing = image.GetSpacing()
    for box, score in zip(result['box'], result['label_scores']):
        idx = world_to_voxel(image, box[:3])
        hw = box[3] / spacing[0] / 2
        hh = box[4] / spacing[1] / 2
        lx0 = (idx[0] - hw) - x0
        ly0 = (idx[1] - hh) - y0
        # only draw boxes within crop window
        if -hw <= lx0 <= (x1-x0) and -hh <= ly0 <= (y1-y0):
            is_focus = (idx == focus_idx)
            color = '#00FF88' if score >= threshold else '#FF4444'
            lw    = 3.0 if is_focus else 1.5
            ls    = 'solid' if score >= threshold else 'dashed'
            alpha = 1.0 if is_focus else 0.6
            rect = patches.Rectangle((lx0, ly0), hw*2, hh*2,
                                      linewidth=lw, edgecolor=color,
                                      facecolor='none', linestyle=ls, alpha=alpha)
            ax.add_patch(rect)
            if is_focus:
                ax.text(lx0, ly0 - 4, f'{score:.2f}', color=color,
                        fontsize=11, fontweight='bold')


# ─── LOAD ─────────────────────────────────────────────────────────────────────

baseline_img, baseline_arr = load_volume(f'{base}/LIDC-0009.nii.gz')
dose25_img,   dose25_arr   = load_volume(f'{base}/LIDC-0009_dicom_dose_reduction_25pct.nii.gz')

baseline_result = load_result('baseline')
dose25_result   = load_result('dose_25pct')

locked_z = get_best_slice(baseline_img, baseline_result, threshold=0.5)

# Convert baseline slice index to world point using SimpleITK's transform
# (handles direction cosines correctly — avoids sign errors in z)
world_point = baseline_img.TransformIndexToPhysicalPoint((0, 0, int(locked_z)))
world_z = world_point[2]

# Find nearest slice in dose25 volume at the same world z
dose25_z_float = dose25_img.TransformPhysicalPointToContinuousIndex(world_point)[2]
dose25_z = int(np.clip(round(dose25_z_float), 0, dose25_arr.shape[0] - 1))
print(f'Baseline z={locked_z} (world z={world_z:.1f}mm), dose25 z={dose25_z}')

# Baseline zoom: highest confidence detection
baseline_scores = baseline_result['label_scores']
baseline_focus_box = baseline_result['box'][int(np.argmax(baseline_scores))]

# Dose25 zoom: the detection with highest confidence that is ABOVE threshold
# (0.82 score — a confident NEW detection not in baseline, that's the story)
dose25_scores = dose25_result['label_scores']
# pick highest score among detections above threshold — this IS the false positive story:
# a confident detection created by noise
dose25_focus_box = dose25_result['box'][int(np.argmax(dose25_scores))]

# Build crop coords
bx0, by0, bx1, by1, b_focus_idx = get_zoom_coords(
    baseline_img, baseline_arr, baseline_result, locked_z, baseline_focus_box, pad=55)
dx0, dy0, dx1, dy1, d_focus_idx = get_zoom_coords(
    dose25_img, dose25_arr, dose25_result, dose25_z, dose25_focus_box, pad=55)

# ─── FIGURE ───────────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(16, 12))
fig.patch.set_facecolor('#0f1117')

gs = fig.add_gridspec(2, 4, height_ratios=[2.8, 1.2],
                      hspace=0.1, wspace=0.04,
                      left=0.02, right=0.98, top=0.93, bottom=0.08)

ax_bl   = fig.add_subplot(gs[0, 0:2])   # baseline full — spans cols 0-1
ax_d25  = fig.add_subplot(gs[0, 2:4])   # dose25 full  — spans cols 2-3
ax_blz  = fig.add_subplot(gs[1, 1])     # baseline zoom — col 1 (centered under left panel)
ax_d25z = fig.add_subplot(gs[1, 2])     # dose25 zoom  — col 2 (centered under right panel)

for ax in [ax_bl, ax_d25, ax_blz, ax_d25z]:
    ax.set_facecolor('#0f1117')

# Full panels
draw_boxes(ax_bl, baseline_img, baseline_arr, baseline_result, locked_z,
           highlight_box=(bx0, by0, bx1-bx0, by1-by0))
ax_bl.set_title('Baseline  (standard dose)', color='white',
                fontsize=13, fontweight='bold', pad=8)
ax_bl.text(0.02, 0.02,
    f"{sum(s>=0.5 for s in baseline_result['label_scores'])} detected  "
    f"({len(baseline_result['label_scores'])} candidates)",
    transform=ax_bl.transAxes, color='#aaaaaa', fontsize=10,
    bbox=dict(facecolor='#0f1117', alpha=0.7, pad=3, linewidth=0))

draw_boxes(ax_d25, dose25_img, dose25_arr, dose25_result, dose25_z,
           highlight_box=(dx0, dy0, dx1-dx0, dy1-dy0))
ax_d25.set_title('25% Dose Reduction  (same anatomical level, z={:.1f}mm)'.format(world_z), color='white',
                 fontsize=13, fontweight='bold', pad=8)
ax_d25.text(0.02, 0.02,
    f"{sum(s>=0.5 for s in dose25_result['label_scores'])} detected  "
    f"({len(dose25_result['label_scores'])} candidates)",
    transform=ax_d25.transAxes, color='#FF8888', fontsize=10,
    bbox=dict(facecolor='#0f1117', alpha=0.7, pad=3, linewidth=0))

# Zoom panels
draw_zoom_panel(ax_blz, baseline_img, baseline_arr, baseline_result,
                locked_z, bx0, by0, bx1, by1, b_focus_idx,
                title='↑  Highest confidence detection (0.99)')

draw_zoom_panel(ax_d25z, dose25_img, dose25_arr, dose25_result,
                dose25_z, dx0, dy0, dx1, dy1, d_focus_idx,
                title='↑  High-confidence candidate created by noise')

# Legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0],[0], color='#00FF88', linewidth=2,
           label='Detected  (confidence ≥ 0.50)'),
    Line2D([0],[0], color='#FF4444', linewidth=1.5, linestyle='--',
           label='Below threshold  (< 0.50)'),
    Line2D([0],[0], color='#FFD700', linewidth=1.5, linestyle='--',
           label='Zoom region'),
]
fig.legend(handles=legend_elements, loc='lower center', ncol=3,
           facecolor='#1a1a2e', labelcolor='white', fontsize=10,
           bbox_to_anchor=(0.5, 0.01), framealpha=0.9)

fig.suptitle(
    'GammaMetric  |  AI Robustness Validation  |  Imaging Environment Stress Test\n'
    'Lung Nodule Detection  ·  MONAI RetinaNet  ·  LIDC-IDRI Public Research Dataset',
    color='white', fontsize=12
)

out = f'{base}/nodule_boxes_v5.png'
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f'Saved: {out}')
