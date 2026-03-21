import os, glob, json
import numpy as np
import SimpleITK as sitk
import xml.etree.ElementTree as ET

DICOM_BASE   = r'C:\Users\Dan\Desktop\gammametric_output\Lung DICOM\manifest-1600709154662\LIDC-IDRI'
OUTPUT_BASE  = r'C:\Users\Dan\Desktop\gammametric_output'
RESULTS_BASE = r'C:\Users\Dan\Desktop\gammametric_output\nodule_results_each'

NS             = 'http://www.nih.gov'
MIN_READERS    = 3
MATCH_DIST_MM  = 15.0
CONF_THRESHOLD = 0.5

CASES = [
    'LIDC-IDRI-0001', 'LIDC-IDRI-0002', 'LIDC-IDRI-0003', 'LIDC-IDRI-0004',
    'LIDC-IDRI-0005', 'LIDC-IDRI-0006', 'LIDC-IDRI-0007', 'LIDC-IDRI-0008',
    'LIDC-IDRI-0009', 'LIDC-IDRI-0010', 'LIDC-IDRI-0011', 'LIDC-IDRI-0012',
    'LIDC-IDRI-0016', 'LIDC-IDRI-0042', 'LIDC-IDRI-0043',
    'LIDC-IDRI-0086', 'LIDC-IDRI-0087', 'LIDC-IDRI-0089', 'LIDC-IDRI-0092',
    'LIDC-IDRI-0093', 'LIDC-IDRI-0151',
]

CONDITIONS = ['baseline', 'dose_25pct', 'dose_50pct', 'thick_3mm', 'thick_5mm']
LABELS = {
    'baseline':   'Baseline',
    'dose_25pct': '25% Dose',
    'dose_50pct': '50% Dose',
    'thick_3mm':  '3mm Thick',
    'thick_5mm':  '5mm Thick',
}

NII_OVERRIDES = {
    'LIDC-IDRI-0009': r'C:\Users\Dan\Desktop\gammametric_output\LIDC-0009.nii.gz',
}


def find_xmls(case_id):
    """Find XML annotation files for a case, excluding other case subdirectories."""
    search_root = os.path.join(DICOM_BASE, case_id)
    xmls = []
    for dirpath, dirnames, filenames in os.walk(search_root):
        dirnames[:] = [d for d in dirnames if not d.startswith('LIDC-IDRI-')]
        for fname in filenames:
            if fname.endswith('.xml'):
                xmls.append(os.path.join(dirpath, fname))
    return xmls


def parse_annotations(case_id):
    xmls = find_xmls(case_id)
    annotations = []
    for xml_path in xmls:
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
        except ET.ParseError:
            continue
        for s, session in enumerate(root.findall(f'.//{{{NS}}}readingSession')):
            for nod in session.findall(f'{{{NS}}}unblindedReadNodule'):
                rois = nod.findall(f'{{{NS}}}roi')
                if not rois:
                    continue
                z_vals = [float(r.find(f'{{{NS}}}imageZposition').text) for r in rois]
                z_mm = sum(z_vals) / len(z_vals)
                mid_roi = rois[len(rois) // 2]
                edges = mid_roi.findall(f'{{{NS}}}edgeMap')
                if not edges:
                    continue
                xs = [int(e.find(f'{{{NS}}}xCoord').text) for e in edges]
                ys = [int(e.find(f'{{{NS}}}yCoord').text) for e in edges]
                annotations.append({
                    'x': sum(xs) / len(xs),
                    'y': sum(ys) / len(ys),
                    'z_mm': z_mm,
                })
    return annotations


def build_consensus(annotations, image, min_readers=2, cluster_dist_mm=15.0):
    spacing = image.GetSpacing()
    used = [False] * len(annotations)
    clusters = []
    for i, ann in enumerate(annotations):
        if used[i]:
            continue
        cluster = [ann]
        used[i] = True
        for j, other in enumerate(annotations):
            if used[j] or i == j:
                continue
            dx = (ann['x'] - other['x']) * spacing[0]
            dy = (ann['y'] - other['y']) * spacing[1]
            dz = ann['z_mm'] - other['z_mm']
            if np.sqrt(dx**2 + dy**2 + dz**2) < cluster_dist_mm:
                cluster.append(other)
                used[j] = True
        if len(cluster) >= min_readers:
            clusters.append({
                'x':            np.mean([a['x'] for a in cluster]),
                'y':            np.mean([a['y'] for a in cluster]),
                'z_mm':         np.mean([a['z_mm'] for a in cluster]),
                'reader_count': len(cluster),
            })
    return clusters


def pixel_to_world(x, y, z_mm, image):
    origin  = image.GetOrigin()
    spacing = image.GetSpacing()
    wx = origin[0] + x * spacing[0]
    wy = origin[1] + y * spacing[1]
    return wx, wy, z_mm


def load_result(case_id, condition):
    path = os.path.join(RESULTS_BASE, f'{case_id}_{condition}', 'result_luna16_fold0.json')
    if not os.path.exists(path):
        return None, None
    with open(path) as f:
        raw = json.load(f)
    if isinstance(raw, dict) and 'value' in raw:
        data = raw['value'][0]
    elif isinstance(raw, list):
        data = raw[0]
    else:
        return None, None
    boxes  = data.get('box', [])
    scores = data.get('label_scores', [])
    if not boxes:
        return [], []
    return boxes, scores


def match_nodules(boxes, scores, consensus, image):
    results = []
    for i, nod in enumerate(consensus):
        wx, wy, wz = pixel_to_world(nod['x'], nod['y'], nod['z_mm'], image)
        best_score, best_dist = None, None
        for box, score in zip(boxes, scores):
            dist = np.sqrt((box[0]-wx)**2 + (box[1]-wy)**2 + (box[2]-wz)**2)
            if dist < MATCH_DIST_MM:
                if best_score is None or score > best_score:
                    best_score, best_dist = score, dist
        results.append({
            'nodule_idx':   i + 1,
            'reader_count': nod['reader_count'],
            'z_mm':         nod['z_mm'],
            'detected':     best_score is not None and best_score >= CONF_THRESHOLD,
            'best_score':   best_score,
            'best_dist_mm': best_dist,
        })
    return results


# ─── MAIN ─────────────────────────────────────────────────────────────────────

all_case_results = {}
summary = {cond: {'detected': 0, 'total': 0} for cond in CONDITIONS}

print(f'{"Case":<22} {"Nodules":<10}', '  '.join(f'{LABELS[c]:<12}' for c in CONDITIONS))
print('-' * 95)

for case_id in CASES:
    nii_path = NII_OVERRIDES.get(case_id,
               os.path.join(OUTPUT_BASE, f'{case_id}.nii.gz'))

    if not os.path.exists(nii_path):
        print(f'{case_id:<22} no NIfTI')
        continue

    image = sitk.ReadImage(nii_path)
    annotations = parse_annotations(case_id)
    if not annotations:
        print(f'{case_id:<22} no annotations found')
        continue

    consensus = build_consensus(annotations, image, min_readers=MIN_READERS)
    if not consensus:
        print(f'{case_id:<22} no consensus nodules ({len(annotations)} annotations)')
        continue

    case_results = {}
    row = f'{case_id:<22} {len(consensus):<10}'
    for cond in CONDITIONS:
        boxes, scores = load_result(case_id, cond)
        if boxes is None:
            row += f'{"n/a":<14}'
            continue
        matches = match_nodules(boxes, scores, consensus, image)
        det = sum(1 for m in matches if m['detected'])
        tot = len(matches)
        summary[cond]['detected'] += det
        summary[cond]['total']    += tot
        case_results[cond] = matches
        row += f'{det}/{tot:<12}'
    print(row)
    all_case_results[case_id] = case_results

print('-' * 95)
row = f'{"TOTAL":<22} {"":<10}'
for cond in CONDITIONS:
    d, t = summary[cond]['detected'], summary[cond]['total']
    pct = 100 * d / t if t else 0
    row += f'{d}/{t} ({pct:.0f}%){"":<3}'
print(row)

out = os.path.join(OUTPUT_BASE, 'consensus_batch_results.json')
with open(out, 'w') as f:
    json.dump(all_case_results, f, indent=2, default=str)
print(f'\nSaved: {out}')
