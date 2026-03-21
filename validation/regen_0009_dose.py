"""
Regenerate dose reduction NIfTIs for LIDC-0009 from baseline,
using physics-informed noise (same as batch_pipeline_v2).
Run once — skips if files already exist.
"""
import numpy as np
import SimpleITK as sitk

RANDOM_SEED = 42
BASE = r'C:\Users\Dan\Desktop\gammametric_output'

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

rng = np.random.default_rng(RANDOM_SEED)

baseline_path = rf'{BASE}\LIDC-0009.nii.gz'
image = sitk.ReadImage(baseline_path)
print(f'Loaded baseline: size={image.GetSize()} spacing={image.GetSpacing()}')

for label, fraction in [('dose_25pct', 0.25), ('dose_50pct', 0.50)]:
    out_path = rf'{BASE}\LIDC-0009_{label}.nii.gz'
    import os
    if os.path.exists(out_path):
        print(f'Already exists: {out_path}')
        continue
    degraded = apply_dose_reduction(image, fraction, rng)
    sitk.WriteImage(degraded, out_path)
    print(f'Written: {out_path}')

print('Done.')
