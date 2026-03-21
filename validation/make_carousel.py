import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as mlines
import SimpleITK as sitk
from reportlab.lib.pagesizes import landscape
from reportlab.pdfgen import canvas as rl_canvas
import os, tempfile

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
    'baseline':  'Baseline  (1.25 mm)',
    'thick_3mm': '3 mm Slice Thickness',
    'thick_5mm': '5 mm Slice Thickness',
}
OUT_PDF = r'C:\Users\Dan\Desktop\gammametric_output\slice_thickness_carousel.pdf'

CONDS       = ['baseline', 'thick_3mm', 'thick_5mm']
VMIN, VMAX  = -1000, 400
CONF_THRESH = 0.5
ZOOM_HALF   = 80
BOX_HI      = '#00e676'
BOX_LO      = '#ff5252'
ZOOM_COL    = '#ffd740'
BG          = '#111111'

# slide size: 1080x1080 points (square, LinkedIn-friendly)
SLIDE_W, SLIDE_H = 1080, 1080


def load_results(path):
    with open(path) as f:
        data = json.load(f)[0]
    return list(zip(data['label_scores'], data['box']))


def world_to_voxel(image, wx, wy, wz):
    return image.TransformPhysicalPointToIndex((float(wx), float(wy), float(wz)))


def get_slice(image, vz):
    arr = sitk.GetArrayFromImage(image)
    vz = int(np.clip(vz, 0, arr.shape[0] - 1))
    return arr[vz].astype(float)


def window(arr):
    return np.clip((arr - VMIN) / (VMAX - VMIN), 0, 1)


# reference nodule
base_dets = sorted(load_results(RESULT_PATHS['baseline']), reverse=True)
_, focus_box = base_dets[2]
fw, fw2, fw3 = focus_box[0], focus_box[1], focus_box[2]

ref_image = sitk.ReadImage(NII_PATHS['baseline'])

tmp_dir = tempfile.mkdtemp()
slide_files = []

# --- slide 0: title ---
fig, ax = plt.subplots(figsize=(10, 10), facecolor=BG)
ax.set_facecolor(BG)
ax.axis('off')
ax.text(0.5, 0.62, 'Slice Thickness &\nLung Nodule Detection AI',
        ha='center', va='center', color='white',
        fontsize=28, fontweight='bold', transform=ax.transAxes, linespacing=1.4)
ax.text(0.5, 0.42, 'How does reconstruction thickness affect\nmodel confidence on the same nodule?',
        ha='center', va='center', color='#aaaaaa',
        fontsize=16, transform=ax.transAxes, linespacing=1.5)
ax.text(0.5, 0.22,
        'MONAI RetinaNet  ·  LIDC-IDRI Dataset\nGammaMetric AI Robustness Validation',
        ha='center', va='center', color='#666666',
        fontsize=12, transform=ax.transAxes, linespacing=1.5)
p = os.path.join(tmp_dir, 'slide_00.png')
plt.savefig(p, dpi=108, bbox_inches='tight', facecolor=BG)
plt.close()
slide_files.append(p)

# --- slides 1-3: one per condition ---
for i, cond in enumerate(CONDS):
    image   = sitk.ReadImage(NII_PATHS[cond])
    spacing = image.GetSpacing()
    dets    = sorted(load_results(RESULT_PATHS[cond]), reverse=True)

    vx, vy, vz = world_to_voxel(image, fw, fw2, fw3)
    arr_shape = sitk.GetArrayFromImage(image).shape
    vz = int(np.clip(vz, 0, arr_shape[0]-1))
    vx = int(np.clip(vx, 0, arr_shape[2]-1))
    vy = int(np.clip(vy, 0, arr_shape[1]-1))

    sl = window(get_slice(image, vz))

    # zoom center clamped
    zx = int(np.clip(vx, ZOOM_HALF, sl.shape[1] - ZOOM_HALF))
    zy = int(np.clip(vy, ZOOM_HALF, sl.shape[0] - ZOOM_HALF))

    fig = plt.figure(figsize=(10, 10), facecolor=BG)
    gs  = fig.add_gridspec(2, 1, height_ratios=[1.6, 1],
                           hspace=0.06, left=0.04, right=0.96,
                           top=0.91, bottom=0.04)

    # full CT
    ax = fig.add_subplot(gs[0])
    ax.set_facecolor(BG)
    ax.imshow(sl, cmap='gray', vmin=0, vmax=1, aspect='equal')

    det_count = 0
    for score, box in dets:
        bvx, bvy, _ = world_to_voxel(image, box[0], box[1], box[2])
        sz    = max(12, int(box[3] / spacing[0]))
        color = BOX_HI if score >= CONF_THRESH else BOX_LO
        ls    = '-' if score >= CONF_THRESH else '--'
        ax.add_patch(patches.Rectangle(
            (bvx - sz//2, bvy - sz//2), sz, sz,
            linewidth=2, edgecolor=color, facecolor='none', linestyle=ls))
        ax.text(bvx - sz//2 + 2, bvy - sz//2 - 5, f'{score:.2f}',
                color=color, fontsize=11, fontfamily='monospace', fontweight='bold',
                bbox=dict(facecolor=BG, alpha=0.55, pad=1, edgecolor='none'))
        if score >= CONF_THRESH:
            det_count += 1

    ax.add_patch(patches.Rectangle(
        (zx - ZOOM_HALF, zy - ZOOM_HALF), ZOOM_HALF*2, ZOOM_HALF*2,
        linewidth=2, edgecolor=ZOOM_COL, facecolor='none', linestyle='--'))

    ax.set_xlim(0, sl.shape[1])
    ax.set_ylim(sl.shape[0], 0)
    ax.axis('off')
    ax.set_title(TITLES[cond], color='white', fontsize=18, pad=8, fontweight='bold')
    ax.text(0.02, 0.04, f'{det_count} detected  ({len(dets)} candidates)',
            transform=ax.transAxes, color='white', fontsize=11, va='bottom')

    # zoom inset
    ax_z = fig.add_subplot(gs[1])
    ax_z.set_facecolor(BG)
    crop = sl[zy-ZOOM_HALF:zy+ZOOM_HALF, zx-ZOOM_HALF:zx+ZOOM_HALF]
    ax_z.imshow(crop, cmap='gray', vmin=0, vmax=1, aspect='equal',
                extent=[zx-ZOOM_HALF, zx+ZOOM_HALF, zy+ZOOM_HALF, zy-ZOOM_HALF])

    for score, box in dets:
        bvx, bvy, _ = world_to_voxel(image, box[0], box[1], box[2])
        if abs(bvx - zx) < ZOOM_HALF and abs(bvy - zy) < ZOOM_HALF:
            sz    = max(10, int(box[3] / spacing[0]))
            color = BOX_HI if score >= CONF_THRESH else BOX_LO
            ls    = '-' if score >= CONF_THRESH else '--'
            ax_z.add_patch(patches.Rectangle(
                (bvx - sz//2, bvy - sz//2), sz, sz,
                linewidth=2.5, edgecolor=color, facecolor='none', linestyle=ls))
            ax_z.text(bvx - sz//2 + 2, bvy - sz//2 - 5, f'{score:.2f}',
                      color=color, fontsize=13, fontfamily='monospace', fontweight='bold',
                      bbox=dict(facecolor=BG, alpha=0.55, pad=1, edgecolor='none'))

    ax_z.set_xlim(zx-ZOOM_HALF, zx+ZOOM_HALF)
    ax_z.set_ylim(zy+ZOOM_HALF, zy-ZOOM_HALF)
    ax_z.axis('off')

    focus_score = next(
        (s for s, box in dets
         if abs(world_to_voxel(image, box[0], box[1], box[2])[0] - vx) < 25
         and abs(world_to_voxel(image, box[0], box[1], box[2])[1] - vy) < 25),
        None)
    label = f'↑ Nodule confidence: {focus_score:.2f}' if focus_score else '↑ Nodule not detected above threshold'
    col   = BOX_HI if (focus_score and focus_score >= CONF_THRESH) else BOX_LO
    ax_z.set_title(label, color=col, fontsize=13, pad=6)

    # header
    fig.text(0.5, 0.965,
             'GammaMetric  |  AI Robustness Validation  |  Imaging Environment Stress Test',
             ha='center', color='#666666', fontsize=9)

    p = os.path.join(tmp_dir, f'slide_{i+1:02d}.png')
    plt.savefig(p, dpi=108, bbox_inches='tight', facecolor=BG)
    plt.close()
    slide_files.append(p)

# --- slide 4: summary ---
scores_by_cond = {}
for cond in CONDS:
    dets = sorted(load_results(RESULT_PATHS[cond]), reverse=True)
    image = sitk.ReadImage(NII_PATHS[cond])
    vx, vy, vz = world_to_voxel(image, fw, fw2, fw3)
    focus_score = next(
        (s for s, box in dets
         if abs(world_to_voxel(image, box[0], box[1], box[2])[0] - vx) < 25
         and abs(world_to_voxel(image, box[0], box[1], box[2])[1] - vy) < 25),
        None)
    scores_by_cond[cond] = focus_score

fig, ax = plt.subplots(figsize=(10, 10), facecolor=BG)
ax.set_facecolor(BG)
ax.axis('off')
ax.text(0.5, 0.88, 'Summary', ha='center', color='white',
        fontsize=24, fontweight='bold', transform=ax.transAxes)

labels = ['1.25 mm\n(Baseline)', '3 mm', '5 mm']
vals   = [scores_by_cond.get(c) for c in CONDS]
colors = [BOX_HI if (v and v >= CONF_THRESH) else BOX_LO for v in vals]
vals_plot = [v if v else 0 for v in vals]

ax2 = fig.add_axes([0.15, 0.38, 0.70, 0.38])
ax2.set_facecolor('#1a1a1a')
bars = ax2.bar(labels, vals_plot, color=colors, width=0.5, edgecolor='none')
ax2.axhline(CONF_THRESH, color=ZOOM_COL, linewidth=1.5, linestyle='--', alpha=0.8)
ax2.text(2.35, CONF_THRESH + 0.02, 'threshold', color=ZOOM_COL, fontsize=10)
ax2.set_ylim(0, 1.05)
ax2.set_ylabel('Detection Confidence', color='#aaaaaa', fontsize=12)
ax2.tick_params(colors='#aaaaaa', labelsize=12)
for spine in ax2.spines.values():
    spine.set_edgecolor('#333333')
ax2.set_facecolor('#1a1a1a')
for bar, val in zip(bars, vals_plot):
    ax2.text(bar.get_x() + bar.get_width()/2, val + 0.02,
             f'{val:.2f}', ha='center', color='white', fontsize=13, fontweight='bold')

ax.text(0.5, 0.28,
        'Confidence on the same confirmed nodule\ndropped ~7× between 1.25 mm and 3 mm reconstruction.',
        ha='center', color='#cccccc', fontsize=14,
        transform=ax.transAxes, linespacing=1.6)
ax.text(0.5, 0.12,
        'gammametric.com',
        ha='center', color='#aaaaaa', fontsize=13, transform=ax.transAxes)

fig.text(0.5, 0.965,
         'GammaMetric  |  AI Robustness Validation  |  Imaging Environment Stress Test',
         ha='center', color='#666666', fontsize=9)

p = os.path.join(tmp_dir, 'slide_04.png')
plt.savefig(p, dpi=108, bbox_inches='tight', facecolor=BG)
plt.close()
slide_files.append(p)

# --- assemble PDF ---
c = rl_canvas.Canvas(OUT_PDF, pagesize=(SLIDE_W, SLIDE_H))
for slide_path in slide_files:
    c.drawImage(slide_path, 0, 0, width=SLIDE_W, height=SLIDE_H,
                preserveAspectRatio=True, anchor='c')
    c.showPage()
c.save()
print(f'Saved: {OUT_PDF}  ({len(slide_files)} slides)')
