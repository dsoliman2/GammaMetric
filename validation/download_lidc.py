import pylidc as pl

# Query for scans with nodules in the 5-10mm range
scans = pl.query(pl.Scan).all()

count = 0
for scan in scans:
    nodules = scan.cluster_annotations()
    for nodule in nodules:
        # Get consensus diameter
        diameter = nodule[0].diameter
        if 5.0 <= diameter <= 10.0:
            print(f"Patient: {scan.patient_id}, Nodule diameter: {diameter:.1f}mm")
            count += 1
            break
    if count >= 5:
        break
