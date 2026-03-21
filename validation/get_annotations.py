import pylidc as pl
import numpy as np

MIN_READERS = 2  # change to 3 to match LUNA16

def get_consensus_nodules(case_id, min_readers=MIN_READERS):
    """
    Return consensus nodules for a case using pylidc.
    Each nodule is a dict with centroid (world mm), diameter, malignancy,
    reader count, and the raw pylidc Annotation objects.
    """
    scan_id = int(case_id.split('-')[-1])
    scans = pl.query(pl.Scan).filter(pl.Scan.patient_id == case_id).all()
    if not scans:
        print(f'  No scan found for {case_id}')
        return []

    scan = scans[0]
    # cluster_annotations groups by proximity; a "cluster" = one physical nodule
    # Returns list of lists: each inner list is annotations from different readers
    nodule_clusters = scan.cluster_annotations()

    consensus = []
    for cluster in nodule_clusters:
        if len(cluster) < min_readers:
            continue

        # Compute centroid in world mm from all reader contours
        all_x, all_y, all_z = [], [], []
        malignancies = []
        diameters = []

        for ann in cluster:
            # centroid in world mm
            cx, cy, cz = ann.centroid(return_LPS=False)  # RAS/ITK convention
            all_x.append(cx)
            all_y.append(cy)
            all_z.append(cz)
            try:
                malignancies.append(ann.malignancy)
            except:
                pass
            try:
                diameters.append(ann.diameter)
            except:
                pass

        consensus.append({
            'centroid_mm': (np.mean(all_x), np.mean(all_y), np.mean(all_z)),
            'reader_count': len(cluster),
            'diameter_mm': np.mean(diameters) if diameters else None,
            'malignancy': np.mean(malignancies) if malignancies else None,
            'annotations': cluster,
        })

    return consensus


if __name__ == '__main__':
    # Quick test on 0009
    case = 'LIDC-IDRI-0009'
    nodules = get_consensus_nodules(case, min_readers=2)
    print(f'{case}: {len(nodules)} consensus nodules (>=2 readers)')
    for i, n in enumerate(nodules):
        cx, cy, cz = n['centroid_mm']
        print(f'  N{i+1}: centroid=({cx:.1f}, {cy:.1f}, {cz:.1f})  '
              f'readers={n["reader_count"]}  '
              f'diam={n["diameter_mm"]:.1f}mm  '
              f'malignancy={n["malignancy"]:.1f}' if n['malignancy'] else
              f'  N{i+1}: centroid=({cx:.1f}, {cy:.1f}, {cz:.1f})  readers={n["reader_count"]}')

    print()
    nodules3 = get_consensus_nodules(case, min_readers=3)
    print(f'{case}: {len(nodules3)} consensus nodules (>=3 readers)')
