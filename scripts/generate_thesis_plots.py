import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style for publication-ready figures
sns.set_theme(style="white", palette="muted")
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Liberation Sans', 'DejaVu Sans'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 14,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

# Path definitions
WORKSPACE = Path("/home/bear/Documents/Workspace/Thesis20252")
MODEL_PATH = WORKSPACE / "engagement-cpu/checkpoints/runs/retrain_deep_forest_repro_balanced_4class_20260626_050152/deep_forest/model.joblib"
FEATURES_DIR = WORKSPACE / "Engagement_DAiSEE/data/processed/features"
OUT_IMP_PATH = WORKSPACE / "Thesis/SOICT_DATN_Base/Hinhve/ket_qua_huan_luyen/09_quan_trong_dac_trung.png"
OUT_CORR_PATH = WORKSPACE / "Thesis/SOICT_DATN_Base/Hinhve/engagement_cpu/preprocess/06_tuong_quan_dac_trung.png"

# Ensure output directories exist
OUT_IMP_PATH.parent.mkdir(parents=True, exist_ok=True)
OUT_CORR_PATH.parent.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------
# PLOT 1: FEATURE IMPORTANCE (LOLLIPOP + DONUT)
# -------------------------------------------------------------
print("Loading model for feature importance...")
model_data = joblib.load(MODEL_PATH)
layer1_models = model_data["layer1"]
et_importances = layer1_models[0].feature_importances_
rf_importances = layer1_models[1].feature_importances_
avg_importances = (et_importances + rf_importances) / 2.0

def get_feature_info(idx):
    if idx == 3528:
        return "Độ dài chuỗi video", "Độ dài chuỗi video", "Đặc trưng Meta", "Độ dài video"
    
    stat_idx = idx // 504
    channel_idx = idx % 504
    
    stats_vi = ['Trung bình', 'Độ lệch chuẩn', 'Tối thiểu', 'Tối đa', 'Khung hình đầu', 'Khung hình cuối', 'Biến thiên (Cuối-Đầu)']
    enrich_vi = ['Gốc', 'Vận tốc', 'Độ lệch chuẩn trượt']
    
    stat_name = stats_vi[stat_idx]
    enrich_name = enrich_vi[channel_idx // 168]
    feat_168_idx = channel_idx % 168
    
    # Categorization
    if feat_168_idx < 25:
        if feat_168_idx in [0, 1, 2]:
            cat = "EAR & MAR"
            name = ['EAR Trái', 'EAR Phải', 'MAR Miệng'][feat_168_idx]
        elif feat_168_idx in [3, 4, 5]:
            cat = "Góc quay đầu (Head Pose)"
            name = ['Pitch (Góc cúi)', 'Yaw (Góc quay)', 'Roll (Góc nghiêng)'][feat_168_idx - 3]
        elif feat_168_idx in range(6, 18):
            cat = "Độ sâu & Tỉ lệ hình học"
            names = [
                'Chiều rộng mặt', 'Chiều cao mặt', 'Khoảng cách mắt', '-Log(Khoảng cách mắt)',
                'Diện tích mặt', 'Tỷ lệ khuôn mặt', 'Độ sâu Z chuẩn hóa', 'Độ sâu Z gốc',
                'Đường kính mống mắt trái', 'Đường kính mống mắt phải', 'Đường kính mống mắt TB', 'Log-nghịch đảo mống mắt'
            ]
            name = names[feat_168_idx - 6]
        elif feat_168_idx in range(18, 24):
            cat = "Quan hệ khoảng cách"
            names = [
                'Mũi-Mắt trái (X)', 'Mũi-Mắt trái (Y)', 'Mũi-Mắt phải (X)', 'Mũi-Mắt phải (Y)',
                'Miệng-Mũi (X)', 'Miệng-Mũi (Y)'
            ]
            name = names[feat_168_idx - 18]
        else:
            cat = "Đặc trưng Meta"
            name = "Độ hợp lệ"
    elif feat_168_idx < 100:
        cat = "Tọa độ 25 điểm mốc (Landmarks)"
        landmark_offset = feat_168_idx - 25
        landmark_idx = landmark_offset // 3
        coord = ['x', 'y', 'z'][landmark_offset % 3]
        ENHANCED_LANDMARKS = [
            1, 4, 10, 33, 46, 52, 61, 78, 93, 133, 152, 159, 168, 172, 197, 234, 263, 276, 282, 291, 308, 323, 362, 386, 454
        ]
        orig_landmark_id = ENHANCED_LANDMARKS[landmark_idx]
        name = f"Điểm mốc {orig_landmark_id} ({coord})"
    elif feat_168_idx < 152:
        cat = "Cơ mặt Blendshape"
        blendshape_idx = feat_168_idx - 100
        name = f"Blendshape {blendshape_idx}"
    else:
        cat = "Ma trận biến đổi Pose"
        matrix_idx = feat_168_idx - 152
        name = f"Ma trận Pose {matrix_idx}"
        
    return stat_name, enrich_name, cat, f"{stat_name} ({enrich_name}) - {name}"

# Aggregate to DataFrame
records = []
for idx in range(len(avg_importances)):
    stat_name, enrich_name, cat, fullname = get_feature_info(idx)
    records.append({
        'index': idx,
        'importance': avg_importances[idx],
        'category': cat,
        'fullname': fullname
    })
df_imp = pd.DataFrame(records)

# Extract top 15 features
top_15 = df_imp.sort_values(by='importance', ascending=False).head(15)

# Calculate category summaries
category_sums = df_imp.groupby('category')['importance'].sum().sort_values(ascending=False)
categories = category_sums.index.tolist()
percentages = (category_sums.values * 100).tolist()

# Define high-end custom color palettes
loll_color = '#1D3557'
donut_colors = ['#1D3557', '#457B9D', '#A8DADC', '#F4A261', '#E76F51', '#2A9D8F', '#E9C46A']

# Create the figure with vertical layout to prevent overlaps and small text
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 11), gridspec_kw={'height_ratios': [1.2, 1]})

# Left Panel: Lollipop plot of Top 15 features
y_ticks = np.arange(len(top_15))
ax1.hlines(y=y_ticks, xmin=0, xmax=top_15['importance'], color='#A8DADC', alpha=0.8, linewidth=2.5)
ax1.scatter(top_15['importance'], y_ticks, color=loll_color, s=100, zorder=3, edgecolors='black', alpha=0.9)

# Labels for top features
ax1.set_yticks(y_ticks)
ax1.set_yticklabels(top_15['fullname'].tolist(), fontsize=11)
ax1.invert_yaxis()  # top importance at the top
ax1.set_xlabel('Mức độ quan trọng (Gini Importance)', fontsize=12, labelpad=8)
ax1.set_title('Top 15 đặc trưng có đóng góp lớn nhất', fontsize=13, weight='bold', pad=15)
ax1.grid(axis='x', linestyle='--', alpha=0.4)
ax1.tick_params(axis='both', which='major', labelsize=11)

# Right Panel: Donut Chart of Category Importances
wedges, texts = ax2.pie(
    percentages,
    labels=None, # Remove inline text labels to avoid overlap
    startangle=140,
    colors=donut_colors[:len(categories)],
    wedgeprops=dict(width=0.4, edgecolor='white', linewidth=2.5)
)

ax2.set_title('Phân bổ đóng góp theo nhóm đặc trưng', fontsize=13, weight='bold', pad=15)
ax2.legend(
    wedges, 
    [f"{c}: {p:.1f}%" for c, p in zip(categories, percentages)],
    title="Nhóm đặc trưng",
    title_fontsize=12,
    loc="center left",
    bbox_to_anchor=(0.95, 0.5),
    fontsize=11,
    frameon=True,
    facecolor='white',
    edgecolor='#E5E7EB'
)

fig.suptitle('Phân tích mức độ quan trọng đặc trưng trong mô hình đề xuất', fontsize=15, weight='bold', y=0.99)
plt.tight_layout()
plt.savefig(OUT_IMP_PATH, bbox_inches='tight', facecolor='white', dpi=300)
plt.close()
print(f"Feature importance plot saved to: {OUT_IMP_PATH}")

# -------------------------------------------------------------
# PLOT 2: CORRELATION HEATMAP (FACIAL METRICS)
# -------------------------------------------------------------
print("Loading DAiSEE features for correlation heatmap...")
npy_files = list(FEATURES_DIR.glob("*.npy"))
if not npy_files:
    print(f"Error: No feature files found in {FEATURES_DIR}")
    exit(1)

# Load a sample subset of features (e.g. 50 files) to get robust correlation matrix
sample_files = npy_files[:80]
data_list = []
for f in sample_files:
    seq = np.load(f)  # shape: (30, 30)
    data_list.append(seq)
combined_data = np.concatenate(data_list, axis=0)  # shape: (N * 30, 30)

# Slice the first 6 features: EAR L, EAR R, MAR, Pitch, Yaw, Roll
facial_metrics = combined_data[:, :6]
df_corr = pd.DataFrame(facial_metrics, columns=[
    'EAR Mắt trái', 'EAR Mắt phải', 'MAR Miệng', 'Pitch (Cúi đầu)', 'Yaw (Quay đầu)', 'Roll (Nghiêng đầu)'
])

# Compute correlation matrix
corr_matrix = df_corr.corr()

# Plot the heatmap
fig, ax = plt.subplots(figsize=(8.5, 7.5))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # show only lower triangle for cleaner layout

sns.heatmap(
    corr_matrix,
    mask=mask,
    cmap='coolwarm',
    vmax=1.0,
    vmin=-1.0,
    center=0,
    annot=True,
    annot_kws={"size": 11},
    fmt=".3f",
    square=True,
    linewidths=1.5,
    cbar_kws={"shrink": 0.8},
    ax=ax
)

# Modify colorbar label and ticks font size
cbar = ax.collections[0].colorbar
cbar.set_label("Hệ số tương quan Pearson", fontsize=12, labelpad=8)
cbar.ax.tick_params(labelsize=11)

ax.set_title('Ma trận tương quan giữa các chỉ số hình học cơ bản', fontsize=13, weight='bold', pad=18)
plt.xticks(rotation=45, ha='right', fontsize=11)
plt.yticks(rotation=0, fontsize=11)
plt.tight_layout()
plt.savefig(OUT_CORR_PATH, bbox_inches='tight', facecolor='white', dpi=300)
plt.close()
print(f"Correlation heatmap saved to: {OUT_CORR_PATH}")
