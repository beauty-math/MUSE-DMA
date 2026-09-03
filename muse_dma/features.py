from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .similarity import (
    gaussian_interaction_profile,
    normalize_similarity,
    similarity_network_fusion,
    topk_normalized_adjacency,
)


def read_matrix(path: str | Path) -> np.ndarray:
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Non-empty matrix file is required: {path}")
    return pd.read_csv(path, sep=r"\s+", header=None, encoding="utf-8-sig").values.astype(np.float32)


def training_association_matrix(train_positive, n_drugs: int, n_microbes: int) -> np.ndarray:
    matrix = np.zeros((n_drugs, n_microbes), dtype=np.float32)
    matrix[
        train_positive["drug_idx0"].to_numpy(dtype=int),
        train_positive["microbe_idx0"].to_numpy(dtype=int),
    ] = 1.0
    return matrix


def _mean_static_similarity(dataset_root: Path, paths: list[str]) -> np.ndarray:
    if not paths:
        raise ValueError("At least one static similarity matrix is required")
    matrices = [normalize_similarity(read_matrix(dataset_root / path)) for path in paths]
    shapes = {matrix.shape for matrix in matrices}
    if len(shapes) != 1:
        raise ValueError(f"Static similarity matrices have incompatible shapes: {sorted(shapes)}")
    return normalize_similarity(np.mean(matrices, axis=0))


def load_semantic_views(dataset_root: Path, embedding_dir: str, keys: list[str]):
    root = dataset_root / embedding_dir
    expected = [root / f"{key}_{entity}.npy" for key in keys for entity in ("drug", "microbe")]
    missing = [str(path) for path in expected if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Semantic embeddings are missing. Run scripts/extract_embeddings.py first:\n"
            + "\n".join(missing)
        )
    drug_views = [np.load(root / f"{key}_drug.npy") for key in keys]
    microbe_views = [np.load(root / f"{key}_microbe.npy") for key in keys]
    if len({view.shape[0] for view in drug_views}) != 1:
        raise ValueError("Drug semantic views contain different entity counts")
    if len({view.shape[0] for view in microbe_views}) != 1:
        raise ValueError("Microbe semantic views contain different entity counts")
    drug = np.hstack(drug_views).astype(np.float32)
    microbe = np.hstack(microbe_views).astype(np.float32)
    return (
        StandardScaler().fit_transform(drug).astype(np.float32),
        StandardScaler().fit_transform(microbe).astype(np.float32),
    )


def build_fold_features(dataset_root: str | Path, config: dict, train_positive, feature_mode: str = "full"):
    dataset_root = Path(dataset_root)
    association = training_association_matrix(
        train_positive, config["n_drugs"], config["n_microbes"]
    )
    drug_static = _mean_static_similarity(dataset_root, config["drug_static_similarities"])
    microbe_static = _mean_static_similarity(dataset_root, config["microbe_static_similarities"])
    if drug_static.shape != (config["n_drugs"], config["n_drugs"]):
        raise ValueError(f"Unexpected drug similarity shape: {drug_static.shape}")
    if microbe_static.shape != (config["n_microbes"], config["n_microbes"]):
        raise ValueError(f"Unexpected microbe similarity shape: {microbe_static.shape}")
    drug_similarity = similarity_network_fusion(
        drug_static, gaussian_interaction_profile(association), k=config["top_k"], steps=config["snf_steps"]
    )
    microbe_similarity = similarity_network_fusion(
        microbe_static, gaussian_interaction_profile(association.T), k=config["top_k"], steps=config["snf_steps"]
    )
    drug_text, microbe_text = load_semantic_views(
        dataset_root, config["embedding_dir"], config["text_keys"]
    )
    if drug_text.shape[0] != config["n_drugs"] or microbe_text.shape[0] != config["n_microbes"]:
        raise ValueError("Semantic embedding entity counts do not match the dataset configuration")

    drug_parts = [drug_similarity, association]
    microbe_parts = [microbe_similarity, association.T]
    if feature_mode != "no_text":
        drug_parts.append(drug_text)
        microbe_parts.append(microbe_text)
    if feature_mode == "text_only":
        drug_parts, microbe_parts = [drug_text], [microbe_text]

    drug_adjacency = topk_normalized_adjacency(drug_similarity, config["top_k"])
    microbe_adjacency = topk_normalized_adjacency(microbe_similarity, config["top_k"])
    if feature_mode in {"no_graph", "text_only"}:
        drug_adjacency = np.eye(config["n_drugs"], dtype=np.float32)
        microbe_adjacency = np.eye(config["n_microbes"], dtype=np.float32)

    return {
        "drug_features": np.hstack(drug_parts).astype(np.float32),
        "microbe_features": np.hstack(microbe_parts).astype(np.float32),
        "drug_adjacency": drug_adjacency,
        "microbe_adjacency": microbe_adjacency,
        "training_association": association,
    }
