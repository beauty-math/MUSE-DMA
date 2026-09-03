from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split


SEED = 20260707


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")


def labeled(pairs, label: int) -> pd.DataFrame:
    if isinstance(pairs, pd.DataFrame):
        frame = pairs[["drug_idx0", "microbe_idx0"]].copy()
    else:
        frame = pd.DataFrame(pairs, columns=["drug_idx0", "microbe_idx0"])
    frame["label"] = label
    return frame


def source_pairs(source: Path) -> tuple[list[str], list[str], pd.DataFrame]:
    drugs = [line for line in (source / "drugs.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    microbes = [line for line in (source / "microbes.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    raw = pd.read_csv(
        source / "adj.txt",
        sep=r"\s+",
        header=None,
        names=["drug_id", "microbe_id", "label"],
    )
    positives = raw.loc[raw.label.eq(1), ["drug_id", "microbe_id"]].drop_duplicates()
    positives.columns = ["drug_idx0", "microbe_idx0"]
    positives = (positives.astype(int) - 1).reset_index(drop=True)
    return drugs, microbes, positives


def unknown_pairs(n_drugs: int, n_microbes: int, positives: pd.DataFrame) -> np.ndarray:
    positive_set = set(map(tuple, positives.values))
    return np.asarray(
        [
            (drug, microbe)
            for drug in range(n_drugs)
            for microbe in range(n_microbes)
            if (drug, microbe) not in positive_set
        ],
        dtype=np.int64,
    )


def rebuild_abiofilm(source: Path, output: Path) -> None:
    drugs, microbes, source_positive = source_pairs(source)
    matrix = np.zeros((len(drugs), len(microbes)), dtype=np.float32)
    matrix[
        source_positive.drug_idx0.to_numpy(), source_positive.microbe_idx0.to_numpy()
    ] = 1.0
    # The final benchmark preserved the public edge-list order before KFold.
    positives = source_positive[["drug_idx0", "microbe_idx0"]].to_numpy(dtype=np.int64)
    negatives_all = np.argwhere(matrix < 0.5)
    rng = np.random.default_rng(SEED)
    negatives = negatives_all[
        rng.choice(len(negatives_all), size=len(positives), replace=False)
    ]
    write_csv(labeled(positives, 1), output / "all_positive_pairs.csv")
    write_csv(
        labeled(negatives, 0),
        output / "all_sampled_negative_pairs_seed20260707.csv",
    )

    positive_folds = list(KFold(5, shuffle=True, random_state=SEED).split(positives))
    negative_folds = list(KFold(5, shuffle=True, random_state=SEED).split(negatives))
    summaries = []
    for fold, ((positive_rest, positive_test), (negative_rest, negative_test)) in enumerate(
        zip(positive_folds, negative_folds), 1
    ):
        positive_train, positive_val = train_test_split(
            positive_rest, test_size=0.10, random_state=SEED + fold
        )
        negative_train, negative_val = train_test_split(
            negative_rest, test_size=0.10, random_state=SEED + fold
        )
        parts = {
            "train_pos": labeled(positives[positive_train], 1),
            "train_neg": labeled(negatives[negative_train], 0),
            "val_pos": labeled(positives[positive_val], 1),
            "val_neg": labeled(negatives[negative_val], 0),
            "test_pos": labeled(positives[positive_test], 1),
            "test_neg": labeled(negatives[negative_test], 0),
        }
        fold_root = output / "folds" / f"fold_{fold}"
        for name, frame in parts.items():
            write_csv(frame, fold_root / f"{name}.csv")
        for split in ["train", "val", "test"]:
            frame = pd.concat(
                [parts[f"{split}_pos"], parts[f"{split}_neg"]], ignore_index=True
            ).sample(
                frac=1, random_state=SEED + fold * 100 + len(split)
            ).reset_index(drop=True)
            write_csv(frame, fold_root / f"{split}.csv")
        summaries.append(
            {
                "fold": fold,
                "train_pos": len(parts["train_pos"]),
                "val_pos": len(parts["val_pos"]),
                "test_pos": len(parts["test_pos"]),
            }
        )
    (output / "split_manifest.json").write_text(
        json.dumps(
            {
                "dataset": "aBiofilm",
                "seed": SEED,
                "n_drugs": len(drugs),
                "n_microbes": len(microbes),
                "n_positive_pairs": len(positives),
                "folds": summaries,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def rebuild_mdad(source: Path, output: Path) -> None:
    drugs, microbes, positives = source_pairs(source)
    unknown = unknown_pairs(len(drugs), len(microbes), positives)
    rng = np.random.default_rng(SEED)
    negatives = pd.DataFrame(
        unknown[rng.choice(len(unknown), size=len(positives), replace=False)],
        columns=["drug_idx0", "microbe_idx0"],
    )
    write_csv(labeled(positives, 1), output / "all_positive_pairs.csv")
    write_csv(
        labeled(negatives, 0),
        output / "all_sampled_negative_pairs_seed20260707.csv",
    )

    summaries = []
    positive_folds = KFold(5, shuffle=True, random_state=SEED)
    negative_folds = KFold(5, shuffle=True, random_state=SEED)
    for fold, ((positive_rest, positive_test), (negative_rest, negative_test)) in enumerate(
        zip(positive_folds.split(positives), negative_folds.split(negatives)), 1
    ):
        positive_train, positive_val = train_test_split(
            positive_rest, test_size=0.10, random_state=SEED + fold
        )
        negative_train, negative_val = train_test_split(
            negative_rest, test_size=0.10, random_state=SEED + fold
        )
        parts = {
            "train_pos": labeled(positives.iloc[positive_train], 1),
            "train_neg": labeled(negatives.iloc[negative_train], 0),
            "val_pos": labeled(positives.iloc[positive_val], 1),
            "val_neg": labeled(negatives.iloc[negative_val], 0),
            "test_pos": labeled(positives.iloc[positive_test], 1),
            "test_neg": labeled(negatives.iloc[negative_test], 0),
        }
        fold_root = output / "folds" / f"fold_{fold}"
        for name, frame in parts.items():
            write_csv(frame, fold_root / f"{name}.csv")
        for split in ["train", "val", "test"]:
            frame = pd.concat(
                [parts[f"{split}_pos"], parts[f"{split}_neg"]], ignore_index=True
            ).sample(
                frac=1, random_state=SEED + fold * 100 + len(split)
            ).reset_index(drop=True)
            write_csv(frame, fold_root / f"{split}.csv")
        summaries.append(
            {
                "fold": fold,
                "train_pos": len(parts["train_pos"]),
                "val_pos": len(parts["val_pos"]),
                "test_pos": len(parts["test_pos"]),
            }
        )
    (output / "split_manifest.json").write_text(
        json.dumps(
            {
                "dataset": "MDAD-2470",
                "seed": SEED,
                "n_drugs": len(drugs),
                "n_microbes": len(microbes),
                "n_positive_pairs": len(positives),
                "folds": summaries,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["abiofilm", "mdad2470"])
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.dataset == "abiofilm":
        rebuild_abiofilm(args.source, args.output)
    else:
        rebuild_mdad(args.source, args.output)
    print(f"Rebuilt {args.dataset} folds in {args.output}")


if __name__ == "__main__":
    main()
