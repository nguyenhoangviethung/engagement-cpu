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
IMG_DIR = WORKSPACE / "Thesis/SOICT_DATN_Base/Hinhve/ket_qua_huan_luyen"
IMG_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ['Rất thấp', 'Thấp', 'Cao', 'Rất cao']
COLORS = ['#1D3557', '#E76F51', '#2A9D8F']

def generate_confusion_matrix():
    # True confusion matrix of Triple XGBoost (calibrated product model)
    cm = np.array([
        [4, 0, 0, 0],
        [0, 36, 6, 3],
        [1, 6, 205, 57],
        [1, 11, 43, 180]
    ], dtype=float)
    
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    annotations = np.empty_like(cm_norm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annotations[i, j] = f'{int(cm[i,j])}\n({cm_norm[i,j]*100:.1f}%)'
            
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    sns.heatmap(cm_norm, annot=annotations, fmt='', cmap='Blues', vmin=0, vmax=1, 
                square=True, cbar_kws={'label': 'Tỷ lệ theo lớp thực'}, ax=ax,
                annot_kws={'fontsize': 11, 'weight': 'bold'})
    
    ax.set_xticklabels(CLASS_NAMES, fontsize=11)
    ax.set_yticklabels(CLASS_NAMES, fontsize=11, rotation=0)
    ax.set_xlabel('Nhãn dự đoán', fontsize=12, labelpad=10)
    ax.set_ylabel('Nhãn thực', fontsize=12, labelpad=10)
    ax.set_title('Ma trận nhầm lẫn của Triple XGBoost trên tập kiểm thử', fontsize=13, weight='bold', pad=15)
    
    out_path = IMG_DIR / '03_ma_tran_nham_lan_triple_xgb_test.png'
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()
    print(f"Confusion matrix saved to: {out_path}")

def generate_per_class_metrics():
    # True metrics of Triple XGBoost per class
    precision = [0.6667, 0.6792, 0.8071, 0.7500]
    recall = [1.0000, 0.8000, 0.7621, 0.7660]
    f1 = [0.8000, 0.7347, 0.7839, 0.7579]
    
    df_list = []
    for idx, cname in enumerate(CLASS_NAMES):
        df_list.append({'Lớp': cname, 'Chỉ số': 'Độ chính xác (Precision)', 'Giá trị': precision[idx]})
        df_list.append({'Lớp': cname, 'Chỉ số': 'Độ bao phủ (Recall)', 'Giá trị': recall[idx]})
        df_list.append({'Lớp': cname, 'Chỉ số': 'Điểm F1-score', 'Giá trị': f1[idx]})
        
    df = pd.DataFrame(df_list)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=df, x='Lớp', y='Giá trị', hue='Chỉ số', palette=COLORS, ax=ax, edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Mức độ tập trung', fontsize=12, labelpad=10)
    ax.set_ylabel('Giá trị chỉ số', fontsize=12, labelpad=10)
    ax.set_title('Độ đo theo từng lớp của Triple XGBoost trên tập kiểm thử', fontsize=13, weight='bold', pad=15)
    ax.yaxis.set_major_formatter(lambda y, pos: f'{y*100:.0f}%')
    ax.set_ylim(0, 1.1)
    
    # Add value labels
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{height*100:.1f}%',
                        xy=(p.get_x() + p.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9.5, fontweight='bold')
            
    ax.legend(title='Chỉ số đánh giá', title_fontsize=11, loc='upper right', frameon=True, facecolor='white', edgecolor='#E5E7EB')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    sns.despine()
    
    out_path = IMG_DIR / '04_do_do_theo_lop_triple_xgb_test.png'
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()
    print(f"Per-class metrics plot saved to: {out_path}")

if __name__ == '__main__':
    generate_confusion_matrix()
    generate_per_class_metrics()
