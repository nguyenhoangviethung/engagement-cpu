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
OUT_PATH = WORKSPACE / "Thesis/SOICT_DATN_Base/Hinhve/ket_qua_huan_luyen/07_so_sanh_thanh_phan_triple_xgb.png"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# Data for Triple XGBoost components and final fused model
data = [
    {
        'Mô hình': 'XGBoost đơn\n(final_xgb)',
        'Độ chính xác': 0.7382,
        'Độ chính xác cân bằng': 0.7635,
        'F1 vĩ mô': 0.7727
    },
    {
        'Mô hình': 'XGBoost tăng cường\n(boost_xgb)',
        'Độ chính xác': 0.8756,
        'Độ chính xác cân bằng': 0.7097,
        'F1 vĩ mô': 0.7867
    },
    {
        'Mô hình': 'XGBoost mục tiêu\n(targeted_xgb)',
        'Độ chính xác': 0.6827,
        'Độ chính xác cân bằng': 0.7128,
        'F1 vĩ mô': 0.7049
    },
    {
        'Mô hình': 'Triple XGBoost\n(Đề xuất - Calibrated)',
        'Độ chính xác': 0.7685,
        'Độ chính xác cân bằng': 0.8320,
        'F1 vĩ mô': 0.7691
    }
]

df = pd.DataFrame(data)
df_melted = df.melt(id_vars='Mô hình', var_name='Chỉ số', value_name='Giá trị')

# Color palette: elegant dark blue, warm rust, and green teal
colors = ['#1D3557', '#E76F51', '#2A9D8F']

fig, ax = plt.subplots(figsize=(10, 6.5))
sns.barplot(data=df_melted, x='Mô hình', y='Giá trị', hue='Chỉ số', palette=colors, ax=ax, edgecolor='black', linewidth=0.5)

# Format axes
ax.set_xlabel('', fontsize=12)
ax.set_ylabel('Kết quả trên tập kiểm thử (Test Set)', fontsize=12, labelpad=10)
ax.set_title('So sánh hiệu năng giữa Triple XGBoost và các mô hình thành phần', fontsize=14, weight='bold', pad=20)
ax.yaxis.set_major_formatter(lambda y, pos: f'{y*100:.0f}%')
ax.set_ylim(0, 1.05)

# Add value labels on top of bars
for p in ax.patches:
    height = p.get_height()
    if height > 0:
        ax.annotate(f'{height*100:.2f}%',
                    xy=(p.get_x() + p.get_width() / 2, height),
                    xytext=(0, 4),  # 4 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9.5, fontweight='bold')

# Customize legend and grid
ax.legend(title='Chỉ số đánh giá', title_fontsize=11, loc='upper left', bbox_to_anchor=(0.01, 0.99), frameon=True, facecolor='white', edgecolor='#E5E7EB', fontsize=10.5)
ax.grid(axis='y', linestyle='--', alpha=0.5)
sns.despine()

plt.tight_layout()
plt.savefig(OUT_PATH, bbox_inches='tight', dpi=300, facecolor='white')
plt.close()
print(f"Component comparison plot saved to: {OUT_PATH}")
