import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as mlines
import SimpleITK as sitk

NII_PATHS = {
    'baseline':  r'C:\Users\Dan\Desktop\gammametric_output\LIDC-0009.nii.gz',
    'thick_3mm': r'C:\Users\Dan\Desktop\gammametric_output\LIDC-0009_thick_3mm.nii.gz',
    'thick_5mm': r'C:\Users\Dan\Desktop\gammametric_output\LIDC-0009_thick_5mm.nii.gz',
}
RESULT_PATHS = {
    'baseline':  r'C:\Users\Dan\Desktop\gammametric_output\nodule_results_each\baseline\result_luna16_fold0.json',
    'thick_3mm': r'C:\Users\Dan\Desktop\gammametric_output\nodule_results_each\thick_3mm\result_luna16_fold0.json',
    'thick_5mm': r'C:\Users\Dan\Desktop\gammametric_output\nodule_results_each\thick_5mm\result_luna16_fold0.json',
}
TITLES = {
    'baseline':  'Baseline  (1.25mm slice thickness)',
    'thick_3mm': '3mm Slice Thickness  (same anatomical level)',
    'thick_5mm': '5mm Slice Thickness  (same anatomical level)',
}
OUT_PATH = r'C:\Users\Dan\Desktop\gammametric_output\slice_thickness_figure_v2.png'

CONDS      = ['baseline', 'thick_3mm', 'thick_5mm']
VMIN, VMAX = -1000, 400
CONF_THRESH = 0.5
ZOOM_HALF   = 60   # half-width of zoom inset in pixels
BOX_COLOR_HI = '#00e676'   # green  — above threshold
BOX_COLOR_LO = '#ff5252'   # red    — below threshold
ZOOM_RECT_COLOR = '#ffd740' # gold dashed zoom indicator


def load_results(path):
    with open(path) as f:
        data = json.load(f)[0]
    return list(zip(data['label_scores'], data['box']))


def world_to_voxel(image, wx, wy, wz):
    return image.TransformPhysicalPointToIndex((float(wx), float(wy), float(wz)))


def get_axial_slice(image, vz):
    arr = sitk.GetArrayFromImage(image)
    vz = int(np.clip(vz, 0, arr.shape[0] - 1))
    return arr[vz].astype(float)


def window(arr):
    return np.clip((arr - VMIN) / (VMAX - VMIN), 0, 1)


# identify focus nodule: 3rd ranked in baseline (most dramatic response)
base_dets = sorted(load_results(RESULT_PATHS['baseline']), reverse=True)
focus_score_base, focus_box = base_dets[2]
fw, fw2, fw3 = focus_box[0], focus_box[1], focus_box[2]

# load baseline image to get reference voxel coords
ref_image = sitk.ReadImage(NII_PATHS['baseline'])
fvx, fvy, fvz = world_to_voxel(ref_image, fw, fw2, fw3)

fig = plt.figure(figsize=(18, 11), facecolor='#111111')
# layout: top row = full CT panels, bottom row = zoom insets
gs = fig.add_gridspec(2, 3, height_ratios=[2.2, 1],
                      hspace=0.08, wspace=0.04,
                      left=0.02, right=0.98, top=0.88, bottom=0.08)

for col, cond in enumerate(CONDS):
    image   = sitk.ReadImage(NII_PATHS[cond])
    spacing = image.GetSpacing()
    dets    = sorted(load_results(RESULT_PATHS[cond]), reverse=True)

    # find the slice closest to focus nodule world z in this resampled image
    # use TransformPhysicalPointToIndex which respects spacing/origin of each volume
    vx, vy, vz = world_to_voxel(image, fw, fw2, fw3)
    # clamp to valid range
    arr_shape = sitk.GetArrayFromImage(image).shape
    vz = int(np.clip(vz, 0, arr_shape[0] - 1))
    vx = int(np.clip(vx, 0, arr_shape[2] - 1))
    vy = int(np.clip(vy, 0, arr_shape[1] - 1))
    sl = window(get_axial_slice(image, vz))

    # --- full CT panel ---
    ax = fig.add_subplot(gs[0, col])
    ax.set_facecolor('#111111')
    ax.imshow(sl, cmap='gray', vmin=0, vmax=1, aspect='equal')

    detected_count = 0
    total_count = len(dets)

    for score, box in dets:
        bvx, bvy, bvz = world_to_voxel(image, box[0], box[1], box[2])
        sz = max(10, int(box[3] / spacing[0]))
        color = BOX_COLOR_HI if score >= CONF_THRESH else BOX_COLOR_LO
        ls    = '-' if score >= CONF_THRESH else '--'
        rect  = patches.Rectangle(
            (bvx - sz//2, bvy - sz//2), sz, sz,
            linewidth=1.5, edgecolor=color, facecolor='none', linestyle=ls
        )
        ax.add_patch(rect)
        ax.text(bvx - sz//2 + 2, bvy - sz//2 - 4,
                f'{score:.2f}',
                color=color, fontsize=8.5, fontfamily='monospace',
                fontweight='bold',
                bbox=dict(facecolor='#111111', alpha=0.5, pad=1, edgecolor='none'))
        if score >= CONF_THRESH:
            detected_count += 1

    # clamp zoom center so full box stays within image
    zx = int(np.clip(vx, ZOOM_HALF, sl.shape[1] - ZOOM_HALF))
    zy = int(np.clip(vy, ZOOM_HALF, sl.shape[0] - ZOOM_HALF))

    zoom_rect = patches.Rectangle(
        (zx - ZOOM_HALF, zy - ZOOM_HALF), ZOOM_HALF*2, ZOOM_HALF*2,
        linewidth=1.8, edgecolor=ZOOM_RECT_COLOR,
        facecolor='none', linestyle='--'
    )
    ax.add_patch(zoom_rect)

    ax.set_xlim(0, sl.shape[1])
    ax.set_ylim(sl.shape[0], 0)
    ax.axis('off')
    ax.set_title(TITLES[cond], color='white', fontsize=11, pad=5, loc='center')

    det_text = f'{detected_count} detected  ({total_count} candidates)'
    ax.text(0.02, 0.03, det_text, transform=ax.transAxes,
            color='white', fontsize=9, va='bottom')

    # --- zoom inset ---
    ax_z = fig.add_subplot(gs[1, col])
    ax_z.set_facecolor('#111111')

    x0 = zx - ZOOM_HALF
    x1 = zx + ZOOM_HALF
    y0 = zy - ZOOM_HALF
    y1 = zy + ZOOM_HALF
    crop = sl[y0:y1, x0:x1]
    ax_z.imshow(crop, cmap='gray', vmin=0, vmax=1, aspect='equal',
                extent=[x0, x1, y1, y0])

    # draw detections within zoom
    for score, box in dets:
        bvx, bvy, bvz = world_to_voxel(image, box[0], box[1], box[2])
        if x0 <= bvx <= x1 and y0 <= bvy <= y1:
            sz = max(8, int(box[3] / spacing[0]))
            color = BOX_COLOR_HI if score >= CONF_THRESH else BOX_COLOR_LO
            ls    = '-' if score >= CONF_THRESH else '--'
            rect  = patches.Rectangle(
                (bvx - sz//2, bvy - sz//2), sz, sz,
                linewidth=2, edgecolor=color, facecolor='none', linestyle=ls
            )
            ax_z.add_patch(rect)
            ax_z.text(bvx - sz//2 + 2, bvy - sz//2 - 4,
                      f'{score:.2f}',
                      color=color, fontsize=9, fontfamily='monospace',
                      fontweight='bold',
                      bbox=dict(facecolor='#111111', alpha=0.5, pad=1, edgecolor='none'))

    ax_z.set_xlim(x0, x1)
    ax_z.set_ylim(y1, y0)
    ax_z.axis('off')

    # find focus nodule score in this condition
    focus_score = None
    for score, box in dets:
        bvx, bvy, _ = world_to_voxel(image, box[0], box[1], box[2])
        if abs(bvx - vx) < 20 and abs(bvy - vy) < 20:
            focus_score = score
            break
    label = f'\u2191 Focus nodule confidence: {focus_score:.2f}' if focus_score else '\u2191 Focus nodule (not detected)'
    color = BOX_COLOR_HI if (focus_score and focus_score >= CONF_THRESH) else BOX_COLOR_LO
    ax_z.set_title(label, color=color, fontsize=9, pad=4)

# header
fig.text(0.5, 0.965,
         'GammaMetric  |  AI Robustness Validation  |  Imaging Environment Stress Test',
         ha='center', color='#aaaaaa', fontsize=9)
fig.text(0.5, 0.945,
         'Lung Nodule Detection  ·  MONAI RetinaNet  ·  LIDC-IDRI Public Research Dataset',
         ha='center', color='#aaaaaa', fontsize=9)

# legend
legend_elements = [
    mlines.Line2D([0], [0], color=BOX_COLOR_HI, linewidth=2,
                  label=f'Detected  (confidence \u2265 {CONF_THRESH})'),
    mlines.Line2D([0], [0], color=BOX_COLOR_LO, linewidth=2, linestyle='--',
                  label=f'Below threshold  (< {CONF_THRESH})'),
    mlines.Line2D([0], [0], color=ZOOM_RECT_COLOR, linewidth=2, linestyle='--',
                  label='Zoom region'),
]
fig.legend(handles=legend_elements, loc='lower center', ncol=3,
           frameon=True, facecolor='#222222', edgecolor='#444444',
           labelcolor='white', fontsize=9, bbox_to_anchor=(0.5, 0.01))

plt.savefig(OUT_PATH, dpi=150, bbox_inches='tight',
            facecolor='#111111', edgecolor='none')
print(f'Saved: {OUT_PATH}')
