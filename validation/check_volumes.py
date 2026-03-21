import SimpleITK as sitk

paths = {
    'baseline':  r'C:\Users\Dan\Desktop\gammametric_output\LIDC-0009.nii.gz',
    'thick_3mm': r'C:\Users\Dan\Desktop\gammametric_output\LIDC-0009_dicom_thick_slices_3mm.nii.gz',
    'thick_5mm': r'C:\Users\Dan\Desktop\gammametric_output\LIDC-0009_dicom_thick_slices_5mm.nii.gz',
}

for name, path in paths.items():
    img = sitk.ReadImage(path)
    print(f'{name}:')
    print(f'  size={img.GetSize()}  spacing={[round(s,2) for s in img.GetSpacing()]}')
    print(f'  origin={[round(o,1) for o in img.GetOrigin()]}')
    # world z of focus detection
    idx = img.TransformPhysicalPointToIndex((80.2, 66.4, -174.2))
    print(f'  voxel for focus nodule world (-174.2mm z): {idx}')
