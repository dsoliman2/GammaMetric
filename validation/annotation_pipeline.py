import os, glob, json
import numpy as np
import pydicom
import xml.etree.ElementTree as ET
import SimpleITK as sitk

DICOM_BASE = r'C:\Users\Dan\Desktop\gammametric_output\Lung DICOM\manifest-1600709154662\LIDC-IDRI'
NS = 'http://www.nih.gov'
MIN_READERS = 2
CLUSTER_DIST_MM = 15.0


def index_dicom_sops(case_dir):
    """Build SOP UID -> (dicom_path, pixel_spacing, image_position) index."""
    sop_index = {}
    for dcm_path in glob.glob(os.path.join(case_dir, '**', '*.dcm'), recursive=True):
        try:
            ds = pydicom.dcmread(dcm_path, stop_before_pixels=True)
            sop = str(ds.SOPInstanceUID)
            spacing = [float(x) for x in ds.PixelSpacing] if hasattr(ds, 'PixelSpacing') else [1.0, 1.0]
            pos = [float(x) for x in ds.ImagePositionPatient] if hasattr(ds, 'ImagePositionPatient') else [0.0, 0.0, 0.0]
            sop_index[sop] = {'path': dcm_path, 'spacing': spacing, 'position': pos}
        except Exception:
            continue
    return sop_index


def parse_xml_annotations(case_dir):
    """Parse all XML files for a case, returning per-reader nodule annotations."""
    xmls = glob.glob(os.path.join(case_dir, '**', '*.xml'), recursive=True)
    all_nodules = []
    for xml_path in xmls:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for reader_idx, session in enumerate(root.findall(f'.//{{{NS}}}readingSession')):
            for nod in session.findall(f'{{{NS}}}unblindedReadNodule'):
                rois = nod.findall(f'{{{NS}}}roi')
                if not rois:
                    continue
                slices = []
                for roi in rois:
                    sop = None
                    sop_elem = roi.find(f'{{{NS}}}imageSOP_UID')
                    if sop_elem is not None and sop_elem.text:
                        sop = sop_elem.text.strip()
                    z = float(roi.find(f'{{{NS}}}imageZposition').text)
                    edges = roi.findall(f'{{{NS}}}edgeMap')
                    xs = [int(e.find(f'{{{NS}}}xCoord').text) for e in edges]
                    ys = [int(e.find(f'{{{NS}}}yCoord').text) for e in edges]
                    if xs and ys:
                        slices.append({
                            'sop': sop,
                            'z_mm': z,
                            'cx_px': np.mean(xs),
                            'cy_px': np.mean(ys),
                        })
                if slices:
                    all_nodules.append({
                        'reader': f'{os.path.basename(xml_path)}_r{reader_idx}',
                        'slices': slices,
                    })
    return all_nodules


def nodule_centroid_mm(nodule, sop_index):
    """
    Compute 3D centroid in world mm for a nodule using SOP UID lookup.
    Falls back to z_mm + pixel spacing from any matched slice if SOP missing.
    """
    wx_list, wy_list, wz_list = [], [], []
    for sl in nodule['slices']:
        sop = sl.get('sop')
        if sop and sop in sop_index:
            info = sop_index[sop]
            ox, oy, oz = info['position']
            sx, sy = info['spacing']
            wx = ox + sl['cx_px'] * sx
            wy = oy + sl['cy_px'] * sy
            wz = oz  # z from ImagePositionPatient is more accurate than XML z
            wx_list.append(wx)
            wy_list.append(wy)
            wz_list.append(wz)
        else:
            # fallback: use z from XML, defer x/y
            wz_list.append(sl['z_mm'])

    if not wx_list:
        # no SOP matches — use first available spacing and XML z
        fallback = next(iter(sop_index.values())) if sop_index else None
        if fallback:
            sx, sy = fallback['spacing']
            mid = nodule['slices'][len(nodule['slices']) // 2]
            ox, oy, _ = fallback['position']
            wx_list = [ox + mid['cx_px'] * sx]
            wy_list = [oy + mid['cy_px'] * sy]

    return (
        np.mean(wx_list) if wx_list else 0.0,
        np.mean(wy_list) if wy_list else 0.0,
        np.mean(wz_list) if wz_list else 0.0,
    )


def cluster_nodules(nodule_centroids, cluster_dist_mm=CLUSTER_DIST_MM):
    """Cluster nodule centroids and return groups with >=MIN_READERS members."""
    used = [False] * len(nodule_centroids)
    clusters = []
    for i, (cx, cy, cz) in enumerate(nodule_centroids):
        if used[i]:
            continue
        group = [i]
        used[i] = True
        for j, (cx2, cy2, cz2) in enumerate(nodule_centroids):
            if used[j]:
                continue
            dist = np.sqrt((cx-cx2)**2 + (cy-cy2)**2 + (cz-cz2)**2)
            if dist < cluster_dist_mm:
                group.append(j)
                used[j] = True
        clusters.append(group)
    return clusters


def get_consensus_nodules(case_id, min_readers=MIN_READERS):
    case_dir = os.path.join(DICOM_BASE, case_id)
    sop_index = index_dicom_sops(case_dir)
    raw_nodules = parse_xml_annotations(case_dir)

    centroids = [nodule_centroid_mm(n, sop_index) for n in raw_nodules]
    clusters = cluster_nodules(centroids)

    consensus = []
    for group in clusters:
        if len(group) < min_readers:
            continue
        cx = np.mean([centroids[i][0] for i in group])
        cy = np.mean([centroids[i][1] for i in group])
        cz = np.mean([centroids[i][2] for i in group])
        consensus.append({
            'centroid_mm': (cx, cy, cz),
            'reader_count': len(group),
        })
    return consensus


if __name__ == '__main__':
    case = 'LIDC-IDRI-0009'
    for min_r in [2, 3]:
        nodules = get_consensus_nodules(case, min_readers=min_r)
        print(f'{case} (>={min_r} readers): {len(nodules)} consensus nodules')
        for i, n in enumerate(nodules):
            cx, cy, cz = n['centroid_mm']
            print(f'  N{i+1}: ({cx:.1f}, {cy:.1f}, {cz:.1f})  readers={n["reader_count"]}')

    print('\nBaseline MONAI detections for comparison:')
    result_path = r'C:\Users\Dan\Desktop\gammametric_output\nodule_results_each\baseline\result_luna16_fold0.json'
    with open(result_path) as f:
        data = json.load(f)[0]
    for box, score in zip(data['box'], data['label_scores']):
        print(f'  ({box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f})  score={score:.3f}')
