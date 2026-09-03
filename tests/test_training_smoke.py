from pathlib import Path

import numpy as np
import pandas as pd

from muse_dma.training import run_cross_validation


def test_one_epoch_training_smoke(tmp_path: Path):
    dataset = tmp_path / "dataset"
    fold = dataset / "folds" / "fold_1"
    embeddings = dataset / "embeddings"
    fold.mkdir(parents=True)
    embeddings.mkdir()
    np.savetxt(dataset / "drug_similarity.txt", np.eye(4), fmt="%.1f")
    np.savetxt(dataset / "microbe_similarity.txt", np.eye(3), fmt="%.1f")
    for key in ["pubmedbert", "biobert", "biolinkbert", "biomedlm"]:
        np.save(embeddings / f"{key}_drug.npy", np.arange(8, dtype=np.float32).reshape(4, 2))
        np.save(embeddings / f"{key}_microbe.npy", np.arange(6, dtype=np.float32).reshape(3, 2))

    def table(rows):
        return pd.DataFrame(rows, columns=["drug_idx0", "microbe_idx0", "label"])

    train_pos = table([(0, 0, 1), (1, 1, 1)])
    train = pd.concat([train_pos, table([(2, 0, 0), (3, 2, 0)])], ignore_index=True)
    val = table([(0, 1, 1), (2, 2, 0)])
    test = table([(3, 1, 1), (1, 2, 0)])
    for name, frame in {"train_pos": train_pos, "train": train, "val": val, "test": test}.items():
        frame.to_csv(fold / f"{name}.csv", index=False)

    config = {
        "dataset": "smoke",
        "n_drugs": 4,
        "n_microbes": 3,
        "drug_static_similarities": ["drug_similarity.txt"],
        "microbe_static_similarities": ["microbe_similarity.txt"],
        "embedding_dir": "embeddings",
        "text_keys": ["pubmedbert", "biobert", "biolinkbert", "biomedlm"],
        "folds": [1],
        "seed": 7,
        "top_k": 2,
        "snf_steps": 1,
        "hidden": 8,
        "dropout": 0.0,
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "epochs": 1,
        "patience": 1,
        "eval_every": 1,
        "threshold": 0.5,
    }
    frame, summary = run_cross_validation(config, dataset, tmp_path / "run")
    assert len(frame) == 1
    assert summary["dataset"] == "smoke"
    assert (tmp_path / "run" / "fold_1" / "test_predictions.csv").is_file()
