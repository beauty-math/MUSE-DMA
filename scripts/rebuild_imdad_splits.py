from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def save_pairs(path: Path, pairs: np.ndarray, label: int) -> pd.DataFrame:
    table = pd.DataFrame(
        {
            "drug_idx0": pairs[:, 0].astype(int),
            "microbe_idx0": pairs[:, 1].astype(int),
            "drug_id1": (pairs[:, 0] + 1).astype(int),
            "microbe_id1": (pairs[:, 1] + 1).astype(int),
            "label": int(label),
        }
    )
    table.to_csv(path, index=False, lineterminator="\n")
    return table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root", type=Path)
    parser.add_argument("output_root", type=Path)
    args = parser.parse_args()

    data = args.data_root
    output = args.output_root
    split_dir = output / "folds"
    split_dir.mkdir(parents=True, exist_ok=True)

    seed = 20260707
    rng = np.random.default_rng(seed)
    n_drugs = sum(
        1
        for _ in open(
            data / "drug/drug_name.txt", encoding="utf-8-sig", errors="ignore"
        )
    )
    n_microbes = sum(
        1
        for _ in open(
            data / "microbe/microbe_name.txt", encoding="utf-8-sig", errors="ignore"
        )
    )

    adj_path = data / "adj/microbe_drug_adj.txt"
    edges_1b = pd.read_csv(
        adj_path, sep=r"\s+", header=None, encoding="utf-8-sig"
    ).astype(int).values
    if edges_1b.shape[1] != 2:
        raise ValueError(f"Unexpected edge-list shape: {edges_1b.shape}")
    positives = np.unique(edges_1b - 1, axis=0)
    if (
        positives[:, 0].min() < 0
        or positives[:, 0].max() >= n_drugs
        or positives[:, 1].min() < 0
        or positives[:, 1].max() >= n_microbes
    ):
        raise ValueError("Association indices are out of range")

    positive_set = {tuple(pair) for pair in positives.tolist()}
    all_pairs = np.array(
        [(drug, microbe) for drug in range(n_drugs) for microbe in range(n_microbes)],
        dtype=int,
    )
    negative_mask = np.array(
        [tuple(pair) not in positive_set for pair in all_pairs], dtype=bool
    )
    negatives = all_pairs[negative_mask]

    rng.shuffle(positives)
    rng.shuffle(negatives)
    positive_folds = np.array_split(positives, 5)
    negative_folds = np.array_split(negatives[: len(positives)], 5)
    save_pairs(output / "all_positive_pairs.csv", positives, 1)
    save_pairs(
        output / "all_sampled_negative_pairs_seed20260707.csv",
        np.concatenate(negative_folds),
        0,
    )

    summaries = []
    for fold_index in range(5):
        fold_root = split_dir / f"fold_{fold_index + 1}"
        fold_root.mkdir(parents=True, exist_ok=True)
        test_pos = positive_folds[fold_index]
        test_neg = negative_folds[fold_index]
        remaining_pos = np.concatenate(
            [positive_folds[j] for j in range(5) if j != fold_index]
        )
        remaining_neg = np.concatenate(
            [negative_folds[j] for j in range(5) if j != fold_index]
        )
        fold_rng = np.random.default_rng(seed + fold_index + 1)
        fold_rng.shuffle(remaining_pos)
        fold_rng.shuffle(remaining_neg)
        n_val_pos = max(1, int(round(len(remaining_pos) * 0.1)))
        n_val_neg = max(1, int(round(len(remaining_neg) * 0.1)))
        val_pos, train_pos = remaining_pos[:n_val_pos], remaining_pos[n_val_pos:]
        val_neg, train_neg = remaining_neg[:n_val_neg], remaining_neg[n_val_neg:]

        parts = {
            "train_pos": (train_pos, 1),
            "val_pos": (val_pos, 1),
            "test_pos": (test_pos, 1),
            "train_neg": (train_neg, 0),
            "val_neg": (val_neg, 0),
            "test_neg": (test_neg, 0),
        }
        loaded = {
            name: save_pairs(fold_root / f"{name}.csv", pairs, label)
            for name, (pairs, label) in parts.items()
        }
        combined = {
            "train": pd.concat(
                [loaded["train_pos"], loaded["train_neg"]], ignore_index=True
            ),
            "val": pd.concat(
                [loaded["val_pos"], loaded["val_neg"]], ignore_index=True
            ),
            "test": pd.concat(
                [loaded["test_pos"], loaded["test_neg"]], ignore_index=True
            ),
        }
        for name, table in combined.items():
            table.sample(frac=1.0, random_state=seed + fold_index).reset_index(
                drop=True
            ).to_csv(fold_root / f"{name}.csv", index=False, lineterminator="\n")

        summaries.append(
            {
                "fold": fold_index + 1,
                "train_pos": len(train_pos),
                "val_pos": len(val_pos),
                "test_pos": len(test_pos),
                "train_neg": len(train_neg),
                "val_neg": len(val_neg),
                "test_neg": len(test_neg),
            }
        )

    manifest = {
        "dataset": "IMDAD",
        "seed": seed,
        "n_drugs": n_drugs,
        "n_microbes": n_microbes,
        "n_positive_pairs": len(positives),
        "n_sampled_negative_pairs": sum(len(part) for part in negative_folds),
        "folds": summaries,
    }
    (output / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
