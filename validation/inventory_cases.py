import os, glob, xml.etree.ElementTree as ET

DICOM_BASE = r'C:\Users\Dan\Desktop\gammametric_output\Lung DICOM\manifest-1600709154662\LIDC-IDRI'
NS = 'http://www.nih.gov'

case_dirs = sorted(glob.glob(os.path.join(DICOM_BASE, 'LIDC-IDRI-*')))
print(f'Found {len(case_dirs)} cases\n')
print(f'{"Case":<20} {"XMLs":<6} {"Nodule annotations":<22} {"2+ reader consensus"}')
print('-' * 70)

for case_dir in case_dirs:
    case_id = os.path.basename(case_dir)
    xmls = glob.glob(os.path.join(case_dir, '**', '*.xml'), recursive=True)
    
    if not xmls:
        print(f'{case_id:<20} 0      -')
        continue

    all_anns = []
    for xml_path in xmls:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for s, session in enumerate(root.findall(f'.//{{{NS}}}readingSession')):
            for nod in session.findall(f'{{{NS}}}unblindedReadNodule'):
                rois = nod.findall(f'{{{NS}}}roi')
                if rois:
                    z_vals = [float(r.find(f'{{{NS}}}imageZposition').text) for r in rois]
                    all_anns.append(round(sum(z_vals)/len(z_vals), 1))

    # rough consensus count: cluster by z within 10mm
    clusters = []
    used = [False]*len(all_anns)
    for i, z in enumerate(all_anns):
        if used[i]: continue
        group = [z]
        used[i] = True
        for j, z2 in enumerate(all_anns):
            if not used[j] and abs(z - z2) < 10:
                group.append(z2)
                used[j] = True
        clusters.append(len(group))

    consensus = sum(1 for c in clusters if c >= 2)
    print(f'{case_id:<20} {len(xmls):<6} {len(all_anns):<22} {consensus}')
