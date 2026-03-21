import os, glob, json, sys, subprocess
import numpy as np
import SimpleITK as sitk

DICOM_BASE   = r'C:\Users\Dan\Desktop\gammametric_output\Lung DICOM\manifest-1600709154662\LIDC-IDRI'
OUTPUT_BASE  = r'C:\Users\Dan\Desktop\gammametric_output'
BUNDLE_DIR   = r'C:\Users\Dan\Desktop\gammametric_output\monai_models\lung_nodule_ct_detection'
RESULTS_BASE = r'C:\Users\Dan\Desktop\gammametric_output\nodule_results_each'

# cases already fully processed (NIfTIs + inference done correctly)
SKIP_CASES = {
    'LIDC-IDRI-0009',  # legacy, done manually
}

# reproducibility
RANDOM_SEED = 42

CONDITIONS = ['baseline', 'dose_25pct', 'dose_50pct', 'thick_3mm', 'thick_5mm']


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
    """
    Physics-informed noise injection.
    CT noise ∝ 1/sqrt(dose), so reducing dose by fraction f
    adds noise: sigma_added = sigma_original * sqrt((1-f)/f)
    where sigma_original estimated from tissue HU std.
    """
    arr = sitk.GetArrayFromImage(image).astype(np.float32)
    tissue_mask = (arr > -500) & (arr < 500)
    sigma_original = np.std(arr[tissue_mask]) if tissue_mask.any() else 20.0
    # noise to add: quadrature combination
    sigma_added = sigma_original * np.sqrt((1.0 - dose_fraction) / dose_fraction)
    noise = rng.normal(0, sigma_added, arr.shape)
    noisy = np.clip(arr + noise, -1024, 3071).astype(np.int16)
    out = sitk.GetImageFromArray(noisy)
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


def make_degraded_versions(image, case_id, rng):
    paths = {}

    # baseline — straight copy of original
    p = os.path.join(OUTPUT_BASE, f'{case_id}_baseline.nii.gz')
    if not os.path.exists(p):
        sitk.WriteImage(image, p)
    paths['baseline'] = p

    # dose reduction — derive from baseline image
    for label, fraction in [('dose_25pct', 0.25), ('dose_50pct', 0.50)]:
        p = os.path.join(OUTPUT_BASE, f'{case_id}_{label}.nii.gz')
        if not os.path.exists(p):
            sitk.WriteImage(apply_dose_reduction(image, fraction, rng), p)
        paths[label] = p

    # thick slices — derive from baseline image
    for label, mm in [('thick_3mm', 3.0), ('thick_5mm', 5.0)]:
        p = os.path.join(OUTPUT_BASE, f'{case_id}_{label}.nii.gz')
        if not os.path.exists(p):
            sitk.WriteImage(apply_thick_slices(image, mm), p)
        paths[label] = p

    return paths


def run_inference(nii_path, case_id, condition):
    out_dir = os.path.join(RESULTS_BASE, f'{case_id}_{condition}')
    result  = os.path.join(out_dir, 'result_luna16_fold0.json')
    if os.path.exists(result):
        print(f'  [skip] {case_id} {condition}')
        return True

    os.makedirs(out_dir, exist_ok=True)
    dl = os.path.join(out_dir, 'datalist.json')
    with open(dl, 'w') as f:
        json.dump({'validation': [{'image': nii_path}]}, f)

    print(f'  [{condition}] running...', end=' ', flush=True)
    proc = subprocess.run(
        [sys.executable, '-m', 'monai.bundle', 'run',
         '--meta_file',           os.path.join(BUNDLE_DIR, 'configs', 'metadata.json'),
         '--config_file',         os.path.join(BUNDLE_DIR, 'configs', 'inference.json'),
         '--bundle_root',         BUNDLE_DIR,
         '--output_dir',          out_dir,
         '--data_list_file_path', dl],
        capture_output=True, text=True,
        cwd=BUNDLE_DIR
    )
    if proc.returncode != 0 or not os.path.exists(result):
        print(f'ERROR')
        print(proc.stderr[-400:])
        return False
    print('done')
    return True


if __name__ == '__main__':
    rng = np.random.default_rng(RANDOM_SEED)

    case_dirs = sorted(glob.glob(os.path.join(DICOM_BASE, 'LIDC-IDRI-*')))
    cases = [os.path.basename(d) for d in case_dirs
             if os.path.basename(d) not in SKIP_CASES]

    print(f'Cases to process: {len(cases)}')

    for case_id in cases:
        print(f'\n=== {case_id} ===')
        dicom_dir = find_largest_ct_series(case_id)
        if not dicom_dir:
            print('  No DICOM found, skipping')
            continue

        nii_path = os.path.join(OUTPUT_BASE, f'{case_id}.nii.gz')
        if os.path.exists(nii_path):
            image = sitk.ReadImage(nii_path)
            print(f'  NIfTI loaded  size={image.GetSize()}')
        else:
            print(f'  Converting DICOM → NIfTI...')
            image = dicom_to_nifti(dicom_dir, nii_path)
            print(f'  Written  size={image.GetSize()}')

        nii_paths = make_degraded_versions(image, case_id, rng)

        for cond in CONDITIONS:
            run_inference(nii_paths[cond], case_id, cond)

    print('\nDone. Run consensus_batch.py to aggregate results.')
