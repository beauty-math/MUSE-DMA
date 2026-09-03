from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


REQUIRED_PAIR_COLUMNS = ["drug_idx0", "microbe_idx0", "label"]


def load_config(path: str | Path) -> dict:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {
        "dataset",
        "n_drugs",
        "n_microbes",
        "drug_static_similarities",
        "microbe_static_similarities",
        "embedding_dir",
        "text_keys",
        "folds",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(f"Configuration lacks keys: {sorted(missing)}")
    return config


def load_fold(dataset_root: str | Path, fold: int):
    fold_root = Path(dataset_root) / "folds" / f"fold_{fold}"
    if not fold_root.is_dir():
        raise FileNotFoundError(f"Frozen fold directory is missing: {fold_root}")
    tables = {}
    for name in ["train", "val", "test", "train_pos"]:
        table = pd.read_csv(fold_root / f"{name}.csv")
        missing = set(REQUIRED_PAIR_COLUMNS) - set(table.columns)
        if missing:
            raise ValueError(f"{name}.csv lacks columns: {sorted(missing)}")
        if table.empty:
            raise ValueError(f"{name}.csv is empty: {fold_root}")
        if not set(table["label"].unique()).issubset({0, 1}):
            raise ValueError(f"{name}.csv contains labels outside {{0, 1}}")
        tables[name] = table
    if not tables["train_pos"]["label"].eq(1).all():
        raise ValueError("train_pos.csv must contain positive pairs only")
    return tables


def read_names(path: str | Path) -> list[str]:
    return [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
