# Model Achievements Summary 4-class

> File kiem tra nhanh cac model 4-class con duoc giu sau cleanup. Bao cao chinh la `bao_cao_ket_qua_huan_luyen_models.md`.

## Topline

| Muc | Model | Accuracy | Balanced Acc | F1 Macro | Model-side latency | Ghi chu |
| :--- | :--- | ---: | ---: | ---: | ---: | :--- |
| Product | `fixed_triple_xgb_fusion` | **76.01%** | **79.98%** | **77.34%** | 11.42 ms | Model product, reproduce bang `scripts/reproduce_product_4class.sh`. |
| Best balanced | `inception_lite_ensemble_balanced` | 71.30% | **80.47%** | 72.45% | 4.20 ms | Balanced cao nhat trong cac model duoc giu. |
| Fastest baseline | `novel_minirocket_4class` | 53.48% | 28.00% | 27.28% | **0.11 ms** | Rat nhe, chi lam baseline latency. |

## Models Kept In Report

| No. | Run | Accuracy | Balanced Acc | F1 Macro | Model-side latency | Report |
| ---: | :--- | ---: | ---: | ---: | ---: | :--- |
| 1 | `fixed_triple_xgb_reproduction` | **76.01%** | **79.98%** | **77.34%** | 11.42 ms | `fixed_triple_xgb_reproduction.json` |
| 2 | `fusion_sweep_xgb_4class` | 74.16% | 77.59% | 76.30% | 11.77 ms | `fusion_sweep_xgb_4class.json` |
| 3 | `daisee_4class_gpu_final` | 73.82% | 76.35% | 77.27% | 0.94 ms | `daisee_4class_gpu_final.json` |
| 4 | `late_fusion_4class_daisee4_fusion` | 73.04% | 76.49% | 76.88% | 435.99 ms | `late_fusion_4class_daisee4_fusion.json` |
| 5 | `inception_lite_ensemble_xgb` | 73.82% | 76.35% | 77.27% | 1.18 ms | `inception_lite_ensemble_xgb.json` |
| 6 | `novel_ordinal_4class` | 73.93% | 62.07% | 69.90% | 5.08 ms | `novel_ordinal_4class.json` |
| 7 | `inception_lite_ensemble_balanced` | 71.30% | **80.47%** | 72.45% | 4.20 ms | `inception_lite_ensemble_balanced.json` |
| 8 | `accuracy_targeted_xgb_4class` | 68.27% | 71.28% | 70.49% | 0.79 ms | `accuracy_targeted_xgb_4class.json` |
| 9 | `novel_minirocket_4class` | 53.48% | 28.00% | 27.28% | 0.11 ms | `novel_minirocket_4class.json` |

## Product Artifacts

| Component | Model | Preprocessor |
| :--- | :--- | :--- |
| `final_xgb` | `checkpoints/runs/product_4class_fixed_triple_xgb/final_xgb/model.json` | `checkpoints/runs/product_4class_fixed_triple_xgb/final_xgb/preprocessor.npz` |
| `boost_xgb` | `checkpoints/runs/product_4class_fixed_triple_xgb/boost_xgb/model.json` | `checkpoints/runs/product_4class_fixed_triple_xgb/boost_xgb/preprocessor.npz` |
| `targeted_xgb` | `checkpoints/runs/product_4class_fixed_triple_xgb/targeted_xgb/model.json` | `checkpoints/runs/product_4class_fixed_triple_xgb/targeted_xgb/preprocessor.npz` |
