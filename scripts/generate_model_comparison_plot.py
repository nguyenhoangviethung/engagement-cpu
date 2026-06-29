import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style for publication-ready figures
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Liberation Sans', 'DejaVu Sans'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.titlesize': 14,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

WORKSPACE = Path("/home/bear/Documents/Workspace/Thesis20252")
OUT_PATH = WORKSPACE / "Thesis/SOICT_DATN_Base/Hinhve/ket_qua_huan_luyen/02_so_sanh_mo_hinh_test.png"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# Data containing SOTA and top-3 trained models only
data = [
    {
        'Mô hình': 'SVD-CNN (SOTA) [23]',
        'Độ chính xác': 0.7797,
        'Độ chính xác cân bằng': 0.7520,
        'F1 vĩ mô': 0.7350,
        'Loại': 'SOTA'
    },
    {
        'Mô hình': 'Triple XGBoost\n(Đề xuất - Calibrated)',
        'Độ chính xác': 0.7685,
        'Độ chính xác cân bằng': 0.8320,
        'F1 vĩ mô': 0.7691,
        'Loại': 'Tự huấn luyện'
    },
    {
        'Mô hình': 'Depth Forest',
        'Độ chính xác': 0.7685,
        'Độ chính xác cân bằng': 0.8590,
        'F1 vĩ mô': 0.7802,
        'Loại': 'Tự huấn luyện'
    },
    {
        'Mô hình': 'XGBoost đơn',
        'Độ chính xác': 0.7382,
        'Độ chính xác cân bằng': 0.7635,
        'F1 vĩ mô': 0.7727,
        'Loại': 'Tự huấn luyện'
    },
    {
        'Mô hình': 'PCA-CNN [23]',
        'Độ chính xác': 0.7288,
        'Độ chính xác cân bằng': 0.7020,
        'F1 vĩ mô': 0.6890,
        'Loại': 'SOTA'
    }
]

df = pd.DataFrame(data)
df_melted = df.melt(id_vars=['Mô hình', 'Loại'], var_name='Chỉ số', value_name='Giá trị')

# Color palette: elegant dark blue, warm rust, and green teal
colors = ['#1D3557', '#E76F51', '#2A9D8F']

fig, ax = plt.subplots(figsize=(11, 7.5))
sns.barplot(
    data=df_melted, 
    y='Mô hình', 
    x='Giá trị', 
    hue='Chỉ số', 
    palette=colors, 
    ax=ax, 
    edgecolor='black', 
    linewidth=0.5
)

# Format axes
ax.set_xlabel('Kết quả trên tập kiểm thử (Test Set)', fontsize=12, labelpad=10)
ax.set_ylabel('', fontsize=12)
ax.set_title('So sánh mô hình đề xuất với các phương pháp đối chứng và SOTA', fontsize=14, weight='bold', pad=20)
ax.xaxis.set_major_formatter(lambda x, pos: f'{x*100:.0f}%')
ax.set_xlim(0, 1.1)

# Add value labels inside/outside bars
for p in ax.patches:
    width = p.get_width()
    if width > 0:
        ax.annotate(f'{width*100:.2f}%',
                    xy=(width, p.get_y() + p.get_height() / 2),
                    xytext=(5, 0),  # 5 points horizontal offset
                    textcoords="offset points",
                    ha='left', va='center', fontsize=9.5, fontweight='bold')

# Customize legend and grid
ax.legend(title='Chỉ số đánh giá', title_fontsize=11, loc='lower right', frameon=True, facecolor='white', edgecolor='#E5E7EB', fontsize=10.5)
ax.grid(axis='x', linestyle='--', alpha=0.5)
sns.despine()

plt.tight_layout()
plt.savefig(OUT_PATH, bbox_inches='tight', dpi=300, facecolor='white')
plt.close()
print(f"Model comparison plot saved to: {OUT_PATH}")
