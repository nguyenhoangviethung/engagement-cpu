# DAiSEE 4-Class Product Artifact

This folder contains the final 4-class artifact package used for the project demo and deployment reference.

## What this package is

- Task: DAiSEE engagement recognition, 4 classes
- Input: processed feature sequence sample from the latest dataset snapshot
- Model: fixed triple-XGBoost fusion
- Use case: CPU-friendly inference and reproducible project artifact

## Included files

- `summary.json`: consolidated evaluation results
- `reproduction_config.json`: reproducible run configuration
- `final_xgb/`: strongest base XGBoost component
- `boost_xgb/`: complementary boosted component
- `targeted_xgb/`: class-targeted component

## Key metrics

- Accuracy: `0.7601`
- Balanced Accuracy: `0.7998`
- Macro F1: `0.7734`
- Model-side latency mean: `11.42 ms`
- E2E latency mean on processed feature sequence sample: `11.37 ms`

## Reproduce locally

Run:

```bash
bash scripts/reproduce_product_4class.sh
```

This will rebuild the product artifact and write the refreshed summary into:

- `checkpoints/runs/product_4class_fixed_triple_xgb/summary.json`

## Notes

- Paths in the package are kept relative for portability.
- The package is intended to be uploaded as a single zip file to Hugging Face Datasets.
- Model-side latency is measured after features are already extracted.
- E2E latency here is measured on a processed feature sequence sample, not on raw video.
