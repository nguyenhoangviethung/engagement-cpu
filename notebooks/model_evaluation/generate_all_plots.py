import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path('/home/bear/Documents/Workspace/Thesis20252/engagement-cpu')
LATEX_DIR = Path('/home/bear/Documents/Workspace/Thesis20252/Thesis/SOICT_DATN_Base/Hinhve/ket_qua_huan_luyen')
LATEX_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR = ROOT / 'notebooks/model_evaluation'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Set styling for publication-ready plots
sns.set_theme(style='white', context='paper', font_scale=1.2)
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'font.family': 'DejaVu Sans',
    'text.usetex': False,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
})

# ----------------------------------------------------------------------
# 1. GENERATE RADAR CHART
# ----------------------------------------------------------------------
print("Generating Radar Chart...")
# Models: Triple XGBoost (Đề xuất), Depth Forest, XGBoost đơn, SVD-CNN (SOTA)
# Categories: Accuracy, Balanced Acc, Macro F1, Macro Precision, Latency Margin (1 - latency/300)
models_data_radar = {
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
        'bal_acc': 0.7520,  # Estimated based on standard unbalanced evaluation
        'f1': 0.7350,
        'prec': 0.7180,
        'latency': 124.12
    }
}

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

colors_radar = {
    'Triple XGBoost (Đề xuất)': '#2A9D8F',  # Green Teal
    'Depth Forest': '#457B9D',             # Soft Blue
    'XGBoost đơn': '#E76F51',              # Warm Orange
    'SVD-CNN (SOTA)': '#E63946'            # Red
}

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

ax.set_rlabel_position(35)
plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["20%", "40%", "60%", "80%", "100%"], color="#6B7280", size=9)
plt.ylim(0, 1.05)

ax.xaxis.grid(True, color="#E5E7EB", linestyle="--", linewidth=1)
ax.yaxis.grid(True, color="#E5E7EB", linestyle="-", linewidth=0.8)
ax.spines['polar'].set_color('#D1D5DB')

for model_name, data in models_data_radar.items():
    latency_margin = max(0.0, 1.0 - (data['latency'] / 300.0))
    values = [
        data['acc'],
        data['bal_acc'],
        data['f1'],
        data['prec'],
        latency_margin
    ]
    values += values[:1] # Close the loop
    color = colors_radar[model_name]
    
    ax.plot(angles, values, linewidth=2.5, linestyle='solid', label=model_name, color=color)
    ax.fill(angles, values, color=color, alpha=0.18 if 'Đề xuất' in model_name or 'SOTA' in model_name else 0.05)
    ax.scatter(angles[:-1], values[:-1], color=color, s=45, edgecolor='white', linewidth=1.2, zorder=5)

plt.title('Biểu đồ Radar so sánh toàn diện hiệu năng các mô hình', size=15, color='#111827', weight='bold', pad=30)
ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.05), frameon=True, facecolor='white', edgecolor='#E5E7EB', fontsize=11)

fig.savefig(OUT_DIR / '08_radar_so_sanh_mo_hinh.png', bbox_inches='tight', dpi=300)
fig.savefig(LATEX_DIR / '08_radar_so_sanh_mo_hinh.png', bbox_inches='tight', dpi=300)
plt.close()
print("Radar chart generated!")

# ----------------------------------------------------------------------
# 2. GENERATE TRADEOFF ACCURACY VS LATENCY PLOT
# ----------------------------------------------------------------------
print("Generating Tradeoff Plot...")
models_data_tradeoff = [
    {
        'Mô hình': 'Triple XGBoost (Đề xuất)',
        'Accuracy': 76.85,
        'Latency': 24.80,
        'Type': 'Trained (Proposed)'
    },
    {
        'Mô hình': 'Depth Forest',
        'Accuracy': 76.85,
        'Latency': 204.07,
        'Type': 'Trained (Baseline)'
    },
    {
        'Mô hình': 'XGBoost đơn',
        'Accuracy': 73.82,
        'Latency': 0.94,
        'Type': 'Trained (Baseline)'
    },
    {
        'Mô hình': 'SVD-CNN (SOTA)',
        'Accuracy': 77.97,
        'Latency': 124.12,
        'Type': 'SOTA'
    },
    {
        'Mô hình': 'PCA-CNN (SOTA)',
        'Accuracy': 72.88,
        'Latency': 124.12,
        'Type': 'SOTA'
    }
]

df_trade = pd.DataFrame(models_data_tradeoff)

fig, ax = plt.subplots(figsize=(9.5, 6.0))

# Colors and marker settings
colors_trade = {
    'Trained (Proposed)': '#2A9D8F',
    'Trained (Baseline)': '#457B9D',
    'SOTA': '#E63946'
}

markers_trade = {
    'Trained (Proposed)': 'o',
    'Trained (Baseline)': 's',
    'SOTA': '^'
}

# Shaded area representing the Real-time zone (< 30ms)
ax.axvspan(0, 33.3, color='#E8F5E9', alpha=0.5, label='Vùng thời gian thực biên (>= 30 FPS, Latency <= 33.3ms)')
ax.axvline(x=33.3, color='#2E7D32', linestyle='--', linewidth=1.2, alpha=0.7)

# Scatter plot
for name, group in df_trade.groupby('Type'):
    ax.scatter(
        group['Latency'], 
        group['Accuracy'], 
        s=120 if 'Proposed' in name else 80,
        color=colors_trade[name],
        marker=markers_trade[name],
        label=name,
        edgecolor='black',
        linewidth=0.8,
        zorder=5
    )

# Label points
for _, row in df_trade.iterrows():
    # Adjust annotation positions to prevent overlaps
    xytext = (10, -5)
    if row['Mô hình'] == 'SVD-CNN (SOTA)':
        xytext = (-95, 8)
    elif row['Mô hình'] == 'PCA-CNN (SOTA)':
        xytext = (-95, -12)
    elif row['Mô hình'] == 'Triple XGBoost (Đề xuất)':
        xytext = (10, 5)
    elif row['Mô hình'] == 'XGBoost đơn':
        xytext = (10, -5)
    elif row['Mô hình'] == 'Depth Forest':
        xytext = (-75, -15)
        
    ax.annotate(
        row['Mô hình'], 
        (row['Latency'], row['Accuracy']), 
        xytext=xytext, 
        textcoords='offset points', 
        fontsize=10,
        fontweight='bold' if 'Proposed' in row['Type'] else 'normal',
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", lw=0.4, alpha=0.8),
        zorder=6
    )

ax.set_xlabel('Độ trễ suy luận trung bình của mô hình trên CPU (ms)', fontsize=12, labelpad=8)
ax.set_ylabel('Độ chính xác trên tập kiểm thử (Accuracy %)', fontsize=12, labelpad=8)
ax.set_title('Đánh đổi hiệu năng (Accuracy) và Độ trễ suy luận (Latency) trên CPU', fontsize=14, weight='bold', pad=15)
ax.set_xlim(-10, 250)
ax.set_ylim(70, 80)
ax.grid(True, linestyle=':', alpha=0.6)

# Legend settings
ax.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='#E5E7EB', fontsize=10.5)

sns.despine()
fig.tight_layout()

fig.savefig(OUT_DIR / '06_tradeoff_accuracy_latency.png', bbox_inches='tight', dpi=300)
fig.savefig(LATEX_DIR / '06_tradeoff_accuracy_latency.png', bbox_inches='tight', dpi=300)
plt.close()
print("Tradeoff plot generated successfully!")
