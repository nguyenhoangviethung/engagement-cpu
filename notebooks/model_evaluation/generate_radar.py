import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path('/home/bear/Documents/Workspace/Thesis20252/engagement-cpu')
ALL_MODELS_PATH = ROOT / 'checkpoints/runs/train_all_4class_gpu_final/train_all_summary.json'
DEEP_FOREST_PATH = ROOT / 'checkpoints/runs/retrain_deep_forest_repro_balanced_4class_20260626_050152/deep_forest/summary.json'

all_models = json.loads(ALL_MODELS_PATH.read_text())
deep_forest = json.loads(DEEP_FOREST_PATH.read_text())

# Set styling
sns.set_theme(style='white', context='paper', font_scale=1.15)
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'font.family': 'DejaVu Sans',
    'text.usetex': False
})

# Define the models we want to compare
# Format: Name, Accuracy, Balanced Accuracy, Macro F1, Macro Precision, Latency (ms)
models_data = {
    'Triple XGBoost (Đề xuất)': {
        'acc': 0.7685,
        'bal_acc': 0.8320,
        'f1': 0.7691,
        'prec': 0.7257,
        'latency': 24.80
    },
    'Depth Forest': {
        'acc': 0.7685,
        'bal_acc': 0.8590,
        'f1': 0.7802,
        'prec': 0.7855,
        'latency': 204.07
    },
    'XGBoost đơn': {
        'acc': 0.7382,
        'bal_acc': 0.7635,
        'f1': 0.7727,
        'prec': 0.7635,
        'latency': 0.94
    },
    'SVD-CNN (SOTA)': {
        'acc': 0.7797,
        'bal_acc': 0.7520,
        'f1': 0.7350,
        'prec': 0.7180,
        'latency': 124.12
    }
}

# Define categories
categories = [
    'Độ chính xác\n(Accuracy)',
    'Độ chính xác cân bằng\n(Balanced Acc)',
    'F1 vĩ mô\n(Macro F1)',
    'Độ chính xác dự đoán\n(Macro Precision)',
    'Biên thời gian thực\n(Latency Margin)'
]
N = len(categories)

# Compute angles for radar chart
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1] # Close the circle

fig, ax = plt.subplots(figsize=(8.5, 8.5), subplot_kw=dict(polar=True))

# Colors for models
colors = {
    'Triple XGBoost (Đề xuất)': '#2A9D8F',        # Green Teal
    'Depth Forest': '#457B9D',                   # Soft Blue
    'XGBoost đơn': '#D95D39',                    # Coral Orange
    'SVD-CNN (SOTA)': '#6D597A'                  # Soft Purple
}

# Draw categories
plt.xticks(angles[:-1], categories, fontsize=11, fontweight='bold', color='#1F2937')
ax.tick_params(axis='x', pad=22)

# Adjust label alignments to avoid overlap
for label, angle in zip(ax.get_xticklabels(), angles[:-1]):
    if angle == 0:
        label.set_horizontalalignment('left')
        label.set_verticalalignment('center')
    elif 0 < angle < np.pi/2:
        label.set_horizontalalignment('left')
        label.set_verticalalignment('bottom')
    elif np.pi/2 <= angle < np.pi:
        label.set_horizontalalignment('right')
        label.set_verticalalignment('bottom')
    elif np.pi <= angle < 3*np.pi/2:
        label.set_horizontalalignment('right')
        label.set_verticalalignment('top')
    else:
        label.set_horizontalalignment('left')
        label.set_verticalalignment('top')

# Draw y-labels (ticks)
ax.set_rlabel_position(35)
plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["20%", "40%", "60%", "80%", "100%"], color="#6B7280", size=9)
plt.ylim(0, 1.05)

# Styling grid lines
ax.xaxis.grid(True, color="#E5E7EB", linestyle="--", linewidth=1)
ax.yaxis.grid(True, color="#E5E7EB", linestyle="-", linewidth=0.8)
ax.spines['polar'].set_color('#D1D5DB')

for model_name, data in models_data.items():
    # Calculate Latency Margin metric: 1 - latency / 300 ms (bounded between 0 and 1)
    latency_margin = max(0.0, 1.0 - (data['latency'] / 300.0))
    
    values = [
        data['acc'],
        data['bal_acc'],
        data['f1'],
        data['prec'],
        latency_margin
    ]
    values += values[:1] # Close the loop
    
    color = colors[model_name]
    
    # Plot line
    ax.plot(angles, values, linewidth=2.5, linestyle='solid', label=model_name, color=color)
    
    # Fill area
    ax.fill(angles, values, color=color, alpha=0.18 if 'Đề xuất' in model_name else 0.08)
    
    # Draw markers at vertices
    ax.scatter(angles[:-1], values[:-1], color=color, s=45, edgecolor='white', linewidth=1.2, zorder=5)

# Add title and legend
plt.title('Biểu đồ Radar so sánh toàn diện hiệu năng các mô hình', size=15, color='#111827', weight='bold', pad=30)
ax.legend(loc='upper right', bbox_to_anchor=(1.22, 1.05), frameon=True, facecolor='white', edgecolor='#E5E7EB', fontsize=10.5)

# Save figure
OUT_DIR = ROOT / 'notebooks/model_evaluation'
LATEX_DIR = Path('/home/bear/Documents/Workspace/Thesis20252/Thesis/SOICT_DATN_Base/Hinhve/ket_qua_huan_luyen')

fig.savefig(OUT_DIR / '08_radar_so_sanh_mo_hinh.png', bbox_inches='tight', dpi=300)
fig.savefig(LATEX_DIR / '08_radar_so_sanh_mo_hinh.png', bbox_inches='tight', dpi=300)
print("Radar chart generated successfully!")
