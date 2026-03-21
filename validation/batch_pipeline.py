import os, glob, json, shutil, subprocess, sys
import numpy as np
import SimpleITK as sitk

DICOM_BASE   = r'C:\Users\Dan\Desktop\gammametric_output\Lung DICOM\manifest-1600709154662\LIDC-IDRI'
OUTPUT_BASE  = r'C:\Users\Dan\Desktop\gammametric_output'
BUNDLE_DIR   = r'C:\Users\Dan\Desktop\gammametric_output\monai_models\lung_nodule_ct_detection'
RESULTS_BASE = r'C:\Users\Dan\Desktop\gammametric_output\nodule_results_each'

# 0009 already done; skip or re-run as needed
CASES = [f'LIDC-IDRI-{str(i).zfill(4)}' for i in range(1, 14)]
SKIP_CASES = {'LIDC-IDRI-0009'}  # already processed

CONDITIONS = ['baseline', 'dose_25pct', 'dose_50pct', 'thick_3mm', 'thick_5mm']


def find_ct_series(case_id):
    """Return DICOM dir for the largest CT series in the case."""
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


def apply_dose_reduction(image, pct):
    arr = sitk.GetArrayFromImage(image).astype(np.float32)
    noise_std = np.std(arr[arr > -900]) * (pct / 100.0) * 0.5
    noise = np.random.normal(0, noise_std, arr.shape)
    noisy = np.clip(arr + noise, -1024, 3071)
    out = sitk.GetImageFromArray(noisy.astype(np.int16))
    out.CopyInformation(image)
    return out


def apply_thick_slices(image, target_mm):
    spacing = list(image.GetSpacing())
    if spacing[2] >= target_mm:
        return image
    factor = int(round(target_mm / spacing[2]))
    arr = sitk.GetArrayFromImage(image)
    slabs = []
    for i in range(0, arr.shape[0] - factor + 1, factor):
        slabs.append(arr[i:i+factor].mean(axis=0))
    thick_arr = np.stack(slabs).astype(np.int16)
    out = sitk.GetImageFromArray(thick_arr)
    new_spacing = (spacing[0], spacing[1], spacing[2] * factor)
    out.SetSpacing(new_spacing)
    out.SetOrigin(image.GetOrigin())
    out.SetDirection(image.GetDirection())
    return out


def make_degraded_versions(image, case_id):
    paths = {}
    base = os.path.join(OUTPUT_BASE, f'{case_id}_baseline.nii.gz')
    sitk.WriteImage(image, base)
    paths['baseline'] = base

    for pct in [25, 50]:
        key = f'dose_{pct}pct'
        p = os.path.join(OUTPUT_BASE, f'{case_id}_{key}.nii.gz')
        sitk.WriteImage(apply_dose_reduction(image, pct), p)
        paths[key] = p

    for mm in [3, 5]:
        key = f'thick_{mm}mm'
        p = os.path.join(OUTPUT_BASE, f'{case_id}_{key}.nii.gz')
        sitk.WriteImage(apply_thick_slices(image, mm), p)
        paths[key] = p

    return paths


def run_monai_inference(nii_path, case_id, condition):
    out_dir = os.path.join(RESULTS_BASE, f'{case_id}_{condition}')
    os.makedirs(out_dir, exist_ok=True)

    datalist_path = os.path.join(out_dir, 'datalist.json')
    with open(datalist_path, 'w') as f:
        json.dump([{'image': nii_path}], f)

    result_path = os.path.join(out_dir, 'result_luna16_fold0.json')
    if os.path.exists(result_path):
        print(f'  [skip] {case_id} {condition} already done')
        return out_dir

    cmd = [
        sys.executable, '-m', 'monai.bundle', 'run',
        '--meta_file', os.path.join(BUNDLE_DIR, 'configs', 'metadata.json'),
        '--config_file', os.path.join(BUNDLE_DIR, 'configs', 'inference.json'),
        '--bundle_root', BUNDLE_DIR,
        '--datalist_file_path', datalist_path,
        '--data_list_key', 'test',
        '--output_dir', out_dir,
    ]

    # patch datalist key — use the same approach that worked for 0009
    # (write a minimal override config)
    override = {'datalist_file_path': datalist_path, 'data_list_key': 'test'}
    ovr_path = os.path.join(out_dir, 'override.json')
    with open(ovr_path, 'w') as f:
        json.dump(override, f)

    cmd = [
        sys.executable, '-m', 'monai.bundle', 'run',
        '--meta_file', os.path.join(BUNDLE_DIR, 'configs', 'metadata.json'),
        '--config_file', os.path.join(BUNDLE_DIR, 'configs', 'inference.json'),
        '--config_file', ovr_path,
        '--bundle_root', BUNDLE_DIR,
        '--output_dir', out_dir,
    ]

    print(f'  Running {case_id} {condition}...')
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f'  ERROR: {result.stderr[-500:]}')
    return out_dir


def load_result(case_id, condition):
    path = os.path.join(RESULTS_BASE, f'{case_id}_{condition}', 'result_luna16_fold0.json')
    if not os.path.exists(path):
        # fallback to old path style (for 0009)
        path = os.path.join(RESULTS_BASE, condition, 'result_luna16_fold0.json')
        if not os.path.exists(path):
            return None, None
    with open(path) as f:
        data = json.load(f)[0]
    return data['box'], data['label_scores']


if __name__ == '__main__':
    for case_id in CASES:
        if case_id in SKIP_CASES:
            print(f'Skipping {case_id} (already done)')
            continue
        print(f'\n=== {case_id} ===')
        dicom_dir = find_ct_series(case_id)
        if not dicom_dir:
            print(f'  No DICOM found, skipping')
            continue
        print(f'  DICOM dir: {dicom_dir}')

        nii_path = os.path.join(OUTPUT_BASE, f'{case_id}.nii.gz')
        if os.path.exists(nii_path):
            print(f'  NIfTI exists, loading...')
            image = sitk.ReadImage(nii_path)
        else:
            print(f'  Converting DICOM → NIfTI...')
            image = dicom_to_nifti(dicom_dir, nii_path)
            print(f'  Written: {nii_path}  size={image.GetSize()}')

        print(f'  Generating degraded versions...')
        nii_paths = make_degraded_versions(image, case_id)

        for cond, path in nii_paths.items():
            run_monai_inference(path, case_id, cond)

    print('\nDone. Run consensus_batch.py to aggregate results.')
