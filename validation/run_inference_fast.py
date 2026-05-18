"""
Fast batch inference — loads the MONAI RetinaNet model ONCE and processes all
cases sequentially. Generates NIfTIs for one case at a time, runs inference,
then deletes them before moving to the next case (keeps disk usage minimal).

Usage:
    python validation/run_inference_fast.py
"""

import os
import sys
import glob
import json
import torch
import numpy as np
import SimpleITK as sitk
from collections import defaultdict

BUNDLE_DIR   = r'C:\Users\Dan\Desktop\gammametric_output\monai_models\lung_nodule_ct_detection'
OUTPUT_BASE  = r'C:\Users\Dan\Desktop\gammametric_output'
RESULTS_BASE = r'C:\Users\Dan\Desktop\gammametric_output\nodule_results_each'
DICOM_BASE   = r'C:\Users\Dan\Desktop\gammametric_output\Lung DICOM\manifest-1600709154662\LIDC-IDRI'

SKIP_CASES = {'LIDC-IDRI-0013'}
RANDOM_SEED = 42

CONDITIONS = ['baseline', 'dose_25pct', 'dose_50pct', 'thick_3mm', 'thick_5mm',
              'soft_kernel', 'sharp_kernel']

INFER_PATCH_SIZE = [512, 512, 192]
OVERLAP          = 0.25
SCORE_THRESH     = 0.02
NMS_THRESH       = 0.22
TOPK             = 1000
DETECTIONS       = 300

sys.path.insert(0, os.path.join(BUNDLE_DIR, 'scripts'))
sys.path.insert(0, BUNDLE_DIR)


# ─── NIFTI GENERATION (from batch_pipeline_v2.py) ────────────────────────────

def find_largest_ct_series(case_id):
    case_dir = os.path.join(DICOM_BASE, case_id)
    best_dir, best_count = None, 0
    for dirpath, _, files in os.walk(case_dir):
        dcms = [f for f in files if f.endswith('.dcm')]
        if len(dcms) > best_count:
            best_count = len(dcms)
            best_dir = dirpath
    return best_dir


def dicom_to_nifti(dicom_dir, out_path):
    reader = sitk.ImageSeriesReader()
    files = reader.GetGDCMSeriesFileNames(dicom_dir)
    reader.SetFileNames(files)
    image = reader.Execute()
    sitk.WriteImage(image, out_path)
    return image


def apply_dose_reduction(image, dose_fraction, rng):
    arr = sitk.GetArrayFromImage(image).astype(np.float32)
    tissue_mask = (arr > -500) & (arr < 500)
    sigma_original = np.std(arr[tissue_mask]) if tissue_mask.any() else 20.0
    sigma_added = sigma_original * np.sqrt((1.0 - dose_fraction) / dose_fraction)
    noise = rng.normal(0, sigma_added, arr.shape)
    noisy = np.clip(arr + noise, -1024, 3071).astype(np.int16)
    out = sitk.GetImageFromArray(noisy)
    out.CopyInformation(image)
    return out


def apply_kvp_reduction(image, rng):
    """
    Simulate 80 kVp acquisition from 120 kVp baseline (image-domain approximation).
    Lower kVp → increased noise + slight HU boost for high-density structures.
    Noise increase factor ~1.5x based on typical 80 vs 120 kVp mAs equivalence.
    """
    arr = sitk.GetArrayFromImage(image).astype(np.float32)
    tissue_mask = (arr > -500) & (arr < 500)
    sigma_original = np.std(arr[tissue_mask]) if tissue_mask.any() else 20.0
    # ~1.5x noise increase for 80 kVp vs 120 kVp at equivalent dose
    sigma_added = sigma_original * np.sqrt(1.5**2 - 1.0)
    noise = rng.normal(0, sigma_added, arr.shape)
    # Slight HU contrast boost for dense structures (bone/calcification)
    contrast_boost = np.where(arr > 100, arr * 0.05, 0.0)
    noisy = np.clip(arr + noise + contrast_boost, -1024, 3071).astype(np.int16)
    out = sitk.GetImageFromArray(noisy)
    out.CopyInformation(image)
    return out


def apply_soft_kernel(image):
    """
    Simulate soft reconstruction kernel (image-domain approximation).
    Soft kernel → Gaussian smoothing, reduces noise but blurs edges.
    sigma=1.0 pixel corresponds to a moderately soft kernel.
    """
    from scipy.ndimage import gaussian_filter
    arr = sitk.GetArrayFromImage(image).astype(np.float32)
    smoothed = gaussian_filter(arr, sigma=1.0).astype(np.int16)
    out = sitk.GetImageFromArray(smoothed)
    out.CopyInformation(image)
    return out


def apply_sharp_kernel(image):
    """
    Simulate sharp reconstruction kernel via unsharp masking.
    Enhances edges/noise texture analogous to a high-frequency kernel (e.g. B60f/lung).
    sigma=0.8 px, strength=1.8 chosen to match pixel_experiment.py calibration.
    """
    from scipy.ndimage import gaussian_filter
    arr = sitk.GetArrayFromImage(image).astype(np.float32)
    blurred = gaussian_filter(arr, sigma=0.8)
    sharpened = np.clip(arr + 1.8 * (arr - blurred), -1024, 3071).astype(np.int16)
    out = sitk.GetImageFromArray(sharpened)
    out.CopyInformation(image)
    return out


def apply_thick_slices(image, target_mm):
    spacing = list(image.GetSpacing())
    if spacing[2] >= target_mm:
        return image
    factor = max(1, int(round(target_mm / spacing[2])))
    arr = sitk.GetArrayFromImage(image).astype(np.float32)
    slabs = []
    for i in range(0, arr.shape[0] - factor + 1, factor):
        slabs.append(arr[i:i+factor].mean(axis=0))
    if not slabs:
        return image
    thick_arr = np.stack(slabs).astype(np.int16)
    out = sitk.GetImageFromArray(thick_arr)
    out.SetSpacing((spacing[0], spacing[1], spacing[2] * factor))
    out.SetOrigin(image.GetOrigin())
    out.SetDirection(image.GetDirection())
    return out


def make_niftis(case_id, rng):
    """Generate NIfTI files for all conditions. Returns dict {condition: path}."""
    dicom_dir = find_largest_ct_series(case_id)
    if not dicom_dir:
        return None

    base_path = os.path.join(OUTPUT_BASE, f'{case_id}.nii.gz')
    if os.path.exists(base_path):
        image = sitk.ReadImage(base_path)
    else:
        image = dicom_to_nifti(dicom_dir, base_path)

    paths = {}

    paths['baseline'] = os.path.join(OUTPUT_BASE, f'{case_id}_baseline.nii.gz')
    if not os.path.exists(paths['baseline']):
        sitk.WriteImage(image, paths['baseline'])

    for label, fraction in [('dose_25pct', 0.25), ('dose_50pct', 0.50)]:
        paths[label] = os.path.join(OUTPUT_BASE, f'{case_id}_{label}.nii.gz')
        if not os.path.exists(paths[label]):
            sitk.WriteImage(apply_dose_reduction(image, fraction, rng), paths[label])

    for label, mm in [('thick_3mm', 3.0), ('thick_5mm', 5.0)]:
        paths[label] = os.path.join(OUTPUT_BASE, f'{case_id}_{label}.nii.gz')
        if not os.path.exists(paths[label]):
            sitk.WriteImage(apply_thick_slices(image, mm), paths[label])

    paths['soft_kernel'] = os.path.join(OUTPUT_BASE, f'{case_id}_soft_kernel.nii.gz')
    if not os.path.exists(paths['soft_kernel']):
        sitk.WriteImage(apply_soft_kernel(image), paths['soft_kernel'])

    paths['sharp_kernel'] = os.path.join(OUTPUT_BASE, f'{case_id}_sharp_kernel.nii.gz')
    if not os.path.exists(paths['sharp_kernel']):
        sitk.WriteImage(apply_sharp_kernel(image), paths['sharp_kernel'])

    return paths


def delete_niftis(case_id):
    for cond in CONDITIONS:
        p = os.path.join(OUTPUT_BASE, f'{case_id}_{cond}.nii.gz')
        try:
            os.remove(p)
        except Exception:
            pass
    base = os.path.join(OUTPUT_BASE, f'{case_id}.nii.gz')
    try:
        os.remove(base)
    except Exception:
        pass


# ─── LOAD MODEL ───────────────────────────────────────────────────────────────

def load_model(device):
    from monai.networks.nets.resnet import resnet50
    from monai.apps.detection.networks.retinanet_network import resnet_fpn_feature_extractor, RetinaNet
    from monai.apps.detection.networks.retinanet_detector import RetinaNetDetector
    from monai.apps.detection.utils.anchor_utils import AnchorGeneratorWithAnchorShape

    backbone = resnet50(spatial_dims=3, n_input_channels=1,
                        conv1_t_stride=[2, 2, 1], conv1_t_size=[7, 7, 7])
    feature_extractor = resnet_fpn_feature_extractor(backbone, 3, False, [1, 2], None)
    network = RetinaNet(
        spatial_dims=3, num_classes=1, num_anchors=3,
        feature_extractor=feature_extractor,
        size_divisible=[16, 16, 8],
        use_list_output=False,
    ).to(device)

    anchor_generator = AnchorGeneratorWithAnchorShape(
        feature_map_scales=[1, 2, 4],
        base_anchor_shapes=[[6, 8, 4], [8, 6, 5], [10, 10, 6]],
    )
    detector = RetinaNetDetector(
        network=network, anchor_generator=anchor_generator,
        debug=False, spatial_dims=3, num_classes=1, size_divisible=[16, 16, 8],
    )
    detector.set_target_keys(box_key='box', label_key='label')
    detector.set_box_selector_parameters(
        score_thresh=SCORE_THRESH, topk_candidates_per_level=TOPK,
        nms_thresh=NMS_THRESH, detections_per_img=DETECTIONS,
    )
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    detector.set_sliding_window_inferer(
        roi_size=INFER_PATCH_SIZE, overlap=OVERLAP,
        sw_batch_size=1, mode='constant', device=device_str,
    )

    ckpt_path = os.path.join(BUNDLE_DIR, 'models', 'model.pt')
    ckpt = torch.load(ckpt_path, map_location='cpu')
    network.load_state_dict(ckpt['model'] if 'model' in ckpt else ckpt)
    network.to(device)
    detector.eval()
    torch.backends.cudnn.benchmark = True
    return detector


# ─── PREPROCESSING / POSTPROCESSING ──────────────────────────────────────────

def build_transform():
    from monai.transforms import (
        Compose, LoadImaged, EnsureChannelFirstd,
        Orientationd, Spacingd, ScaleIntensityRanged, EnsureTyped,
    )
    return Compose([
        LoadImaged(keys='image'),
        EnsureChannelFirstd(keys='image'),
        Orientationd(keys='image', axcodes='RAS'),
        Spacingd(keys='image', pixdim=[0.703125, 0.703125, 1.25]),
        ScaleIntensityRanged(keys='image', a_min=-1024.0, a_max=300.0,
                             b_min=0.0, b_max=1.0, clip=True),
        EnsureTyped(keys='image'),
    ])


def postprocess(pred, image_tensor):
    from monai.apps.detection.transforms.dictionary import (
        ClipBoxToImaged, AffineBoxToWorldCoordinated, ConvertBoxModed,
    )
    data = {**pred, 'image': image_tensor}
    data = ClipBoxToImaged(box_keys='box', label_keys='label',
                           box_ref_image_keys='image', remove_empty=True)(data)
    data = AffineBoxToWorldCoordinated(box_keys='box', box_ref_image_keys='image',
                                       affine_lps_to_ras=True)(data)
    data = ConvertBoxModed(box_keys='box', src_mode='xyzxyz', dst_mode='cccwhd')(data)
    return data


# ─── RUN ONE IMAGE ────────────────────────────────────────────────────────────

def run_one(nii_path, out_dir, result_path, detector, transform, device):
    os.makedirs(out_dir, exist_ok=True)
    data = transform({'image': nii_path})
    image_tensor = data['image'].unsqueeze(0).to(device)

    from contextlib import nullcontext
    amp_ctx = torch.amp.autocast('cuda') if device.type == 'cuda' else nullcontext()
    with amp_ctx, torch.no_grad():
        pred = detector([image_tensor[0]], use_inferer=True)
        pred = pred[0]

    processed = postprocess(pred, data['image'])
    boxes  = processed.get('box', torch.zeros((0, 6))).tolist()
    scores = processed.get('label_scores', torch.zeros(0)).tolist()
    labels = processed.get('label', torch.zeros(0, dtype=torch.long)).tolist()

    with open(result_path, 'w') as f:
        json.dump([{'box': boxes, 'label': labels,
                    'label_scores': scores, 'image': nii_path}], f, indent=2)


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    rng = np.random.default_rng(RANDOM_SEED)

    case_dirs = sorted(glob.glob(os.path.join(DICOM_BASE, 'LIDC-IDRI-*')))
    all_cases = [os.path.basename(d) for d in case_dirs
                 if os.path.basename(d) not in SKIP_CASES]

    # Find cases with incomplete results
    pending_cases = []
    for case_id in all_cases:
        missing = [c for c in CONDITIONS if not os.path.exists(
            os.path.join(RESULTS_BASE, f'{case_id}_{c}', 'result_luna16_fold0.json'))]
        if missing:
            pending_cases.append((case_id, missing))

    total_cases = len(pending_cases)
    total_runs  = sum(len(m) for _, m in pending_cases)
    print(f'Cases with pending inference: {total_cases} ({total_runs} runs)')

    if not pending_cases:
        print('Nothing to do.')
        return

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print('Loading model...')
    detector = load_model(device)
    transform = build_transform()
    print('Model loaded. Starting inference.\n')

    errors = []
    run_num = 0

    for case_num, (case_id, missing_conds) in enumerate(pending_cases, 1):
        print(f'\n[Case {case_num}/{total_cases}] {case_id} — generating NIfTIs...')
        try:
            nii_paths = make_niftis(case_id, rng)
        except Exception as e:
            print(f'  NIfTI generation failed: {e}')
            errors.append((case_id, f'NIfTI error: {e}'))
            continue

        if nii_paths is None:
            print(f'  No DICOM found, skipping.')
            continue

        for cond in missing_conds:
            run_num += 1
            nii_path    = nii_paths[cond]
            out_dir     = os.path.join(RESULTS_BASE, f'{case_id}_{cond}')
            result_path = os.path.join(out_dir, 'result_luna16_fold0.json')
            print(f'  [{run_num}/{total_runs}] {cond} ... ', end='', flush=True)
            try:
                run_one(nii_path, out_dir, result_path, detector, transform, device)
                print('done')
            except Exception as e:
                import traceback
                print(f'ERROR: {e}')
                if len(errors) == 0:
                    print(traceback.format_exc())
                errors.append((f'{case_id}_{cond}', str(e)))

        delete_niftis(case_id)
        print(f'  NIfTIs deleted.')

    print(f'\nFinished. {total_runs - len(errors)} succeeded, {len(errors)} failed.')
    if errors:
        print('Errors:')
        for label, err in errors:
            print(f'  {label}: {err}')


if __name__ == '__main__':
    main()
