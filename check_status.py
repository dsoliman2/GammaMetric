# save as check_status.py and run: python check_status.py
import os, glob

base    = r'C:\Users\Dan\Desktop\gammametric_output'
results = r'C:\Users\Dan\Desktop\gammametric_output\nodule_results_each'

niftis = sorted(glob.glob(os.path.join(base, 'LIDC-IDRI-*.nii.gz')))
cases  = sorted(set(os.path.basename(f).split('_')[0] for f in niftis))
print(f'Cases with NIfTIs: {len(cases)}')
for c in cases:
    print(f'  {c}')

print()
result_dirs = sorted(glob.glob(os.path.join(results, 'LIDC-IDRI-*')))
print(f'Inference result folders: {len(result_dirs)}')
for d in result_dirs:
    has_result = os.path.exists(os.path.join(d, 'result_luna16_fold0.json'))
    status = 'OK' if has_result else 'MISSING'
    print(f'  {os.path.basename(d)}  {status}')