import os, sys, json, glob, subprocess

BUNDLE_DIR   = r'C:\Users\Dan\Desktop\gammametric_output\monai_models\lung_nodule_ct_detection'
OUTPUT_BASE  = r'C:\Users\Dan\Desktop\gammametric_output'
RESULTS_BASE = r'C:\Users\Dan\Desktop\gammametric_output\nodule_results_each'
DICOM_BASE   = r'C:\Users\Dan\Desktop\gammametric_output\Lung DICOM\manifest-1600709154662\LIDC-IDRI'

CONDITIONS = {
    'baseline':   lambda p: p,
    'dose_25pct': lambda p: p.replace('.nii.gz', '_dose_25pct.nii.gz').replace(
                            'LIDC-IDRI-', 'LIDC-IDRI-').replace('\\LIDC-IDRI-', '\\LIDC-IDRI-'),
    'dose_50pct': None,
    'thick_3mm':  None,
    'thick_5mm':  None,
}

# Build list of (case_id, condition, nii_path) jobs
jobs = []
for case_dir in sorted(glob.glob(os.path.join(DICOM_BASE, 'LIDC-IDRI-*'))):
    case_id = os.path.basename(case_dir)

    # skip 0009 legacy
    if case_id == 'LIDC-IDRI-0009':
        continue

    nii_paths = {
        'baseline':   os.path.join(OUTPUT_BASE, f'{case_id}_baseline.nii.gz'),
        'dose_25pct': os.path.join(OUTPUT_BASE, f'{case_id}_dose_25pct.nii.gz'),
        'dose_50pct': os.path.join(OUTPUT_BASE, f'{case_id}_dose_50pct.nii.gz'),
        'thick_3mm':  os.path.join(OUTPUT_BASE, f'{case_id}_thick_3mm.nii.gz'),
        'thick_5mm':  os.path.join(OUTPUT_BASE, f'{case_id}_thick_5mm.nii.gz'),
    }

    for cond, nii_path in nii_paths.items():
        out_dir = os.path.join(RESULTS_BASE, f'{case_id}_{cond}')
        result  = os.path.join(out_dir, 'result_luna16_fold0.json')

        if os.path.exists(result):
            continue  # already done
        if not os.path.exists(nii_path):
            print(f'  MISSING NIfTI: {nii_path}')
            continue

        jobs.append((case_id, cond, nii_path, out_dir))

print(f'{len(jobs)} inference jobs to run\n')

for i, (case_id, cond, nii_path, out_dir) in enumerate(jobs):
    os.makedirs(out_dir, exist_ok=True)
    dl = os.path.join(out_dir, 'datalist.json')
    with open(dl, 'w') as f:
        json.dump({'validation': [{'image': nii_path}]}, f)

    print(f'[{i+1}/{len(jobs)}] {case_id} {cond}...')
    result = subprocess.run(
        [sys.executable, '-m', 'monai.bundle', 'run',
         '--meta_file',          os.path.join(BUNDLE_DIR, 'configs', 'metadata.json'),
         '--config_file',        os.path.join(BUNDLE_DIR, 'configs', 'inference.json'),
         '--bundle_root',        BUNDLE_DIR,
         '--output_dir',         out_dir,
         '--data_list_file_path', dl],
        capture_output=True, text=True,
        cwd=BUNDLE_DIR  # run from bundle root so scripts/ is on path
    )
    if result.returncode != 0:
        print(f'  ERROR: {result.stderr[-300:]}')
    elif os.path.exists(os.path.join(out_dir, 'result_luna16_fold0.json')):
        print(f'  done')
    else:
        print(f'  completed but no result file found')

print('\nAll jobs finished.')
