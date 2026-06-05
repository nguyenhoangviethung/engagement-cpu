from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler

LOGGER = logging.getLogger(__name__)


def _setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")


def _iter_frame_batches(manifest: pd.DataFrame, batch_rows: int):
    buffer: list[np.ndarray] = []
    total = 0
    for row in manifest.itertuples(index=False):
        arr = np.load(Path(row.feature_path)).astype(np.float32)
        buffer.append(arr.reshape(-1, arr.shape[-1]))
        total += buffer[-1].shape[0]
        if total >= batch_rows:
            yield np.concatenate(buffer, axis=0)
            buffer = []
            total = 0
    if buffer:
        yield np.concatenate(buffer, axis=0)


def compress_manifest(manifest_path: Path, output_root: Path, components: list[int], batch_rows: int = 20000) -> dict[str, object]:
    manifest = pd.read_csv(manifest_path)
    required = {"feature_path", "label", "split"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"manifest is missing required columns: {sorted(missing)}")
    feature_dims = set(manifest.get("feature_dim", pd.Series(dtype=int)).dropna().astype(int).tolist())
    if feature_dims and feature_dims != {709}:
        raise ValueError(f"expected openface709 manifest, got feature_dim values: {sorted(feature_dims)}")

    train_manifest = manifest[manifest["split"].astype(str).str.lower() == "train"].reset_index(drop=True)
    if train_manifest.empty:
        raise ValueError("manifest has no train rows; cannot fit train-only scaler/PCA")

    output_root.mkdir(parents=True, exist_ok=True)
    models_dir = output_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Fitting StandardScaler on train frames | rows=%d", len(train_manifest))
    scaler = StandardScaler()
    for batch in _iter_frame_batches(train_manifest, batch_rows=batch_rows):
        scaler.partial_fit(batch)
    joblib.dump(scaler, models_dir / "standard_scaler.joblib")

    max_components = max(components)
    LOGGER.info("Fitting IncrementalPCA | max_components=%d", max_components)
    ipca = IncrementalPCA(n_components=max_components, batch_size=batch_rows)
    for batch in _iter_frame_batches(train_manifest, batch_rows=batch_rows):
        ipca.partial_fit(scaler.transform(batch))
    joblib.dump(ipca, models_dir / "incremental_pca_max.joblib")

    results: dict[str, object] = {
        "source_manifest": str(manifest_path),
        "output_root": str(output_root),
        "components": {},
        "explained_variance_ratio_cumsum": np.cumsum(ipca.explained_variance_ratio_).tolist(),
    }

    for dim in components:
        dim_root = output_root / f"pca{dim}"
        features_dir = dim_root / "features"
        features_dir.mkdir(parents=True, exist_ok=True)
        out_rows = []
        LOGGER.info("Writing compressed features | dim=%d | rows=%d", dim, len(manifest))
        for idx, row in enumerate(manifest.itertuples(index=False), start=1):
            src = Path(row.feature_path)
            arr = np.load(src).astype(np.float32)
            flat = arr.reshape(-1, arr.shape[-1])
            transformed = ipca.transform(scaler.transform(flat))[:, :dim].reshape(arr.shape[0], dim).astype(np.float32)
            out_path = features_dir / src.name
            np.save(out_path, transformed)
            payload = row._asdict()
            payload["feature_path"] = str(out_path)
            payload["source_feature_path"] = str(src)
            payload["source_feature_dim"] = int(arr.shape[-1])
            payload["feature_dim"] = int(dim)
            payload["feature_set"] = f"openface709_pca{dim}"
            out_rows.append(payload)
            if idx % 5000 == 0 or idx == len(manifest):
                LOGGER.info("pca%d progress %d/%d", dim, idx, len(manifest))
        out_manifest = dim_root / "feature_manifest.csv"
        pd.DataFrame(out_rows).to_csv(out_manifest, index=False)
        variance = float(np.sum(ipca.explained_variance_ratio_[:dim]))
        metadata = {
            "source_manifest": str(manifest_path),
            "manifest": str(out_manifest),
            "feature_set": f"openface709_pca{dim}",
            "components": dim,
            "explained_variance_ratio_sum": variance,
            "scaler": str(models_dir / "standard_scaler.joblib"),
            "pca": str(models_dir / "incremental_pca_max.joblib"),
        }
        (dim_root / "compression_summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        results["components"][str(dim)] = metadata

    summary_path = output_root / "compression_summary.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    LOGGER.info("Saved compression summary to %s", summary_path)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train-only PCA compression for OpenFace709 sequence manifests.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--components", type=str, default="128,192,256,300,384")
    parser.add_argument("--batch-rows", type=int, default=20000)
    return parser.parse_args()


def main() -> None:
    _setup_logging()
    args = parse_args()
    components = [int(x.strip()) for x in args.components.split(",") if x.strip()]
    if sorted(set(components)) != components:
        components = sorted(set(components))
    compress_manifest(args.manifest, args.output_root, components, batch_rows=args.batch_rows)


if __name__ == "__main__":
    main()
