from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


DATASET_DIRS = ["IMDAD", "aBiofilm", "MDAD2470"]
ROLES = ["train", "val", "test", "train_pos", "val_pos", "test_pos", "train_neg", "val_neg", "test_neg"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pair_set(frame: pd.DataFrame) -> set[tuple[int, int]]:
    return set(map(tuple, frame[["drug_idx0", "microbe_idx0"]].astype(int).to_numpy()))


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit frozen five-fold pair files")
    parser.add_argument("--repository-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.repository_root.resolve()
    records = []
    for dataset_name in DATASET_DIRS:
        dataset = root / "data" / dataset_name
        positives = pair_set(pd.read_csv(dataset / "processed" / "positive_associations.csv"))
        for fold in range(1, 6):
            fold_root = dataset / "folds" / f"fold_{fold}"
            tables = {}
            for role in ROLES:
                path = fold_root / f"{role}.csv"
                if not path.is_file() or path.stat().st_size == 0:
                    raise RuntimeError(f"Missing or empty split file: {path}")
                table = pd.read_csv(path)
                if not {"drug_idx0", "microbe_idx0", "label"}.issubset(table.columns):
                    raise RuntimeError(f"Required split columns are absent: {path}")
                tables[role] = table
                records.append(
                    {
                        "dataset": dataset_name,
                        "fold": fold,
                        "role": role,
                        "rows": len(table),
                        "positives": int(table.label.eq(1).sum()),
                        "negatives": int(table.label.eq(0).sum()),
                        "sha256": sha256(path),
                    }
                )
            train_pairs, val_pairs, test_pairs = map(pair_set, (tables["train"], tables["val"], tables["test"]))
            if train_pairs & val_pairs or train_pairs & test_pairs or val_pairs & test_pairs:
                raise RuntimeError(f"Pair leakage in {dataset_name} fold {fold}")
            if pair_set(tables["train_pos"]) - positives:
                raise RuntimeError(f"Unknown training positive in {dataset_name} fold {fold}")
            if pair_set(tables["test_pos"]) - positives:
                raise RuntimeError(f"Unknown test positive in {dataset_name} fold {fold}")
            if pair_set(tables["test_neg"]) & positives:
                raise RuntimeError(f"Known positive appears among test negatives in {dataset_name} fold {fold}")
            for split in ("train", "val", "test"):
                expected = pair_set(tables[f"{split}_pos"]) | pair_set(tables[f"{split}_neg"])
                if pair_set(tables[split]) != expected:
                    raise RuntimeError(f"Combined {split} pairs do not match role files in {dataset_name} fold {fold}")
    frame = pd.DataFrame(records)
    frame.to_csv(root / "data" / "frozen_split_manifest.csv", index=False, lineterminator="\n")
    report = {
        "status": "PASS",
        "datasets": len(DATASET_DIRS),
        "folds_per_dataset": 5,
        "audited_files": len(frame),
        "fold_safe_rule": "association-derived features are constructed from train_pos.csv only",
    }
    (root / "data" / "split_audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
