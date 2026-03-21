import json

data = json.load(open(r'C:\Users\Dan\Desktop\gammametric_output\consensus_batch_results.json'))

conditions = ['baseline', 'dose_25pct', 'dose_50pct', 'thick_3mm', 'thick_5mm']
labels = ['Baseline', '25% Dose', '50% Dose', '3mm Thick', '5mm Thick']

agg = {c: {'detected': 0, 'total': 0} for c in conditions}

for case, case_data in data.items():
    for cond in conditions:
        nodules = case_data[cond]
        agg[cond]['detected'] += sum(1 for n in nodules if n['detected'])
        agg[cond]['total'] += len(nodules)

print(f"{'Condition':<15} {'Detected':>10} {'Total':>8} {'Sensitivity':>12}")
print("-" * 48)
for cond, label in zip(conditions, labels):
    det = agg[cond]['detected']
    tot = agg[cond]['total']
    sens = det/tot*100
    print(f"{label:<15} {det:>10} {tot:>8} {sens:>11.1f}%")