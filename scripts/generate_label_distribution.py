import numpy as np
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
OUT_PATH = WORKSPACE / "Thesis/SOICT_DATN_Base/Hinhve/ket_qua_huan_luyen/05_phan_bo_lop_train_test.png"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# Consistent label names matching DAiSEE definition:
# 0=very low, 1=low, 2=high, 3=very high
# Vietnamese: Rất thấp, Thấp, Cao, Rất cao
CLASS_NAMES = ['Rất thấp', 'Thấp', 'Cao', 'Rất cao']

# Approximate counts from DAiSEE dataset (engagement label)
# Train set
train_counts = [346, 2088, 19898, 20358]
# Test set  
test_counts = [4, 45, 269, 235]

x = np.arange(len(CLASS_NAMES))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, train_counts, width, label='Huấn luyện', color='#1D3557', edgecolor='black', linewidth=0.5)
bars2 = ax.bar(x + width/2, test_counts, width, label='Kiểm thử', color='#E76F51', edgecolor='black', linewidth=0.5)

ax.set_xlabel('Mức độ tập trung', fontsize=12, labelpad=10)
ax.set_ylabel('Số video', fontsize=12, labelpad=10)
ax.set_title('Phân bố lớp của tập huấn luyện và tập kiểm thử', fontsize=14, weight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(CLASS_NAMES, fontsize=11)
ax.set_yscale('log')
ax.legend(fontsize=11, frameon=True, facecolor='white', edgecolor='#E5E7EB')
ax.grid(axis='y', linestyle='--', alpha=0.5)
sns.despine()

plt.tight_layout()
plt.savefig(OUT_PATH, bbox_inches='tight', dpi=300, facecolor='white')
plt.close()
print(f"Label distribution plot saved to: {OUT_PATH}")
