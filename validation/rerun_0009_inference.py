"""
Rerun inference for LIDC-0009 dose conditions using corrected NIfTIs.
Run AFTER regen_0009_dose.py.
"""
import os, sys, json, subprocess

BASE        = r'C:\Users\Dan\Desktop\gammametric_output'
BUNDLE_DIR  = r'C:\Users\Dan\Desktop\gammametric_output\monai_models\lung_nodule_ct_detection'
RESULTS_BASE = r'C:\Users\Dan\Desktop\gammametric_output\nodule_results_each'

JOBS = [
    ('baseline',   rf'{BASE}\LIDC-0009.nii.gz',            'LIDC-IDRI-0009_baseline'),
    ('dose_25pct', rf'{BASE}\LIDC-0009_dose_25pct.nii.gz', 'LIDC-IDRI-0009_dose_25pct'),
    ('dose_50pct', rf'{BASE}\LIDC-0009_dose_50pct.nii.gz', 'LIDC-IDRI-0009_dose_50pct'),
    ('thick_3mm',  rf'{BASE}\LIDC-0009_thick_3mm.nii.gz',  'LIDC-IDRI-0009_thick_3mm'),
    ('thick_5mm',  rf'{BASE}\LIDC-0009_thick_5mm.nii.gz',  'LIDC-IDRI-0009_thick_5mm'),
]

for cond, nii_path, result_folder in JOBS:
    out_dir = os.path.join(RESULTS_BASE, result_folder)
    result  = os.path.join(out_dir, 'result_luna16_fold0.json')

    if os.path.exists(result):
        print(f'[skip] {cond} already done at {out_dir}')
        continue

    os.makedirs(out_dir, exist_ok=True)
    dl = os.path.join(out_dir, 'datalist.json')
    with open(dl, 'w') as f:
        json.dump({'validation': [{'image': nii_path}]}, f)

    print(f'[{cond}] running inference...', end=' ', flush=True)
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
        print('ERROR')
        print(proc.stderr[-400:])
    else:
        print('done')

print('\nAll done. Now update fix_comparison.py result paths to use LIDC-IDRI-0009_* folders.')
