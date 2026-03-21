import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Load results
with open(r'C:\Users\Dan\Desktop\gammametric_output\consensus_batch_results.json') as f:
    data = json.load(f)

conditions = ['baseline', 'dose_25pct', 'dose_50pct', 'thick_3mm', 'thick_5mm']
labels = ['Baseline', '25% Dose', '50% Dose', '3mm Thick', '5mm Thick']
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

colors = ['#2C3E50', '#2980B9', '#1ABC9C', '#E67E22', '#C0392B']

# Build sensitivity vs threshold matrix
# sens_matrix[condition_idx][threshold_idx] = sensitivity
sens_matrix = np.zeros((len(conditions), len(thresholds)))

for j, threshold in enumerate(thresholds):
    for i, cond in enumerate(conditions):
        detected = 0
        total = 0
        for case, case_data in data.items():
            for nodule in case_data[cond]:
                total += 1
                score = nodule.get('best_score')
                dist = nodule.get('best_dist_mm')
                # Detected if score >= threshold AND within 15mm
                if score is not None and score >= threshold and dist is not None and dist <= 15.0:
                    detected += 1
        sens_matrix[i, j] = detected / total * 100 if total > 0 else 0

# Print table
print(f"{'Threshold':<12}", end='')
for label in labels:
    print(f"{label:>12}", end='')
print()
print('-' * (12 + 12 * len(labels)))
for j, thresh in enumerate(thresholds):
    print(f"{thresh:<12.1f}", end='')
    for i in range(len(conditions)):
        print(f"{sens_matrix[i,j]:>11.1f}%", end='')
    print()

# Plot
fig, ax = plt.subplots(figsize=(9, 5.5))

for i, (label, color) in enumerate(zip(labels, colors)):
    lw = 2.5 if label == '5mm Thick' else 1.5
    ls = '-' if label in ['Baseline', '5mm Thick'] else '--'
    ax.plot(thresholds, sens_matrix[i], color=color, linewidth=lw,
            linestyle=ls, marker='o', markersize=5, label=label)

ax.set_xlabel('Confidence Threshold', fontsize=12, labelpad=8)
ax.set_ylabel('Sensitivity (%)', fontsize=12, labelpad=8)
ax.set_title('Detection Sensitivity vs. Confidence Threshold\nAcross CT Acquisition Conditions',
             fontsize=13, fontweight='bold', pad=15)
ax.set_xlim(0.05, 0.95)
ax.set_ylim(0, 75)
ax.set_xticks(thresholds)
ax.yaxis.grid(True, linestyle='--', alpha=0.4)
ax.set_axisbelow(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(loc='upper right', fontsize=10, framealpha=0.9)

# Annotate 0.5 threshold
ax.axvline(x=0.5, color='gray', linestyle=':', linewidth=1.2, alpha=0.7)
ax.text(0.505, 68, 'threshold\nused', fontsize=8, color='gray', va='top')

plt.tight_layout()
plt.savefig(r'C:\Users\Dan\Desktop\gammametric_output\fig3_threshold_sensitivity.png',
            dpi=180, bbox_inches='tight')
plt.close()
print("\nFigure saved to Desktop/gammametric_output/fig3_threshold_sensitivity.png")
