from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from .data import load_fold
from .features import build_fold_features
from .metrics import binary_metrics
from .model import MUSEDMA
from .reproducibility import set_seed


def _pair_tensors(table, device):
    pairs = torch.tensor(
        table[["drug_idx0", "microbe_idx0"]].to_numpy(), dtype=torch.long, device=device
    )
    labels = torch.tensor(table["label"].to_numpy(), dtype=torch.float32, device=device)
    return pairs, labels


def train_fold(config: dict, dataset_root: Path, fold: int, output_root: Path, feature_mode: str = "full"):
    set_seed(config["seed"] + fold)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split = load_fold(dataset_root, fold)
    features = build_fold_features(dataset_root, config, split["train_pos"], feature_mode)

    drug_features = torch.tensor(features["drug_features"], device=device)
    microbe_features = torch.tensor(features["microbe_features"], device=device)
    drug_adjacency = torch.tensor(features["drug_adjacency"], device=device)
    microbe_adjacency = torch.tensor(features["microbe_adjacency"], device=device)
    train_pairs, train_labels = _pair_tensors(split["train"], device)
    val_pairs, val_labels = _pair_tensors(split["val"], device)
    test_pairs, test_labels = _pair_tensors(split["test"], device)

    model = MUSEDMA(
        drug_features.shape[1], microbe_features.shape[1],
        hidden=config["hidden"], dropout=config["dropout"]
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    positive_weight = torch.tensor(
        [(train_labels == 0).sum().item() / max(1, (train_labels == 1).sum().item())],
        device=device,
    )

    best_aupr, best_state, best_epoch, bad_epochs = -1.0, None, 0, 0
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        optimizer.zero_grad()
        drug, microbe = model.encode_entities(
            drug_features, microbe_features, drug_adjacency, microbe_adjacency
        )
        logits = model.score_pairs(drug, microbe, train_pairs)
        loss = F.binary_cross_entropy_with_logits(logits, train_labels, pos_weight=positive_weight)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        if epoch == 1 or epoch % config["eval_every"] == 0:
            model.eval()
            with torch.no_grad():
                drug, microbe = model.encode_entities(
                    drug_features, microbe_features, drug_adjacency, microbe_adjacency
                )
                val_scores = torch.sigmoid(model.score_pairs(drug, microbe, val_pairs)).cpu().numpy()
            val_metrics = binary_metrics(val_labels.cpu().numpy(), val_scores, config["threshold"])
            if val_metrics["aupr"] > best_aupr:
                best_aupr = val_metrics["aupr"]
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                best_epoch = epoch
                bad_epochs = 0
            else:
                bad_epochs += config["eval_every"]
                if bad_epochs >= config["patience"]:
                    break

    if best_state is None:
        raise RuntimeError("No checkpoint was selected")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        drug, microbe = model.encode_entities(
            drug_features, microbe_features, drug_adjacency, microbe_adjacency
        )
        val_scores = torch.sigmoid(model.score_pairs(drug, microbe, val_pairs)).cpu().numpy()
        test_scores = torch.sigmoid(model.score_pairs(drug, microbe, test_pairs)).cpu().numpy()

    result = {
        "dataset": config["dataset"],
        "seed": config["seed"] + fold,
        "feature_mode": feature_mode,
        "fold": fold,
        "best_epoch": best_epoch,
        "validation": binary_metrics(val_labels.cpu().numpy(), val_scores, config["threshold"]),
        "test": binary_metrics(test_labels.cpu().numpy(), test_scores, config["threshold"]),
    }
    fold_root = output_root / f"fold_{fold}"
    fold_root.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, fold_root / "model.pt")
    prediction = split["test"][["drug_idx0", "microbe_idx0", "label"]].copy()
    prediction["score"] = test_scores
    prediction.to_csv(fold_root / "test_predictions.csv", index=False)
    (fold_root / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def run_cross_validation(config: dict, dataset_root: Path, output_root: Path, feature_mode: str = "full"):
    output_root.mkdir(parents=True, exist_ok=True)
    results = [
        train_fold(config, dataset_root, fold, output_root, feature_mode)
        for fold in config["folds"]
    ]
    rows = []
    for result in results:
        row = {"fold": result["fold"], "best_epoch": result["best_epoch"]}
        for split_name in ["validation", "test"]:
            for metric, value in result[split_name].items():
                row[f"{split_name}_{metric}"] = value
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(output_root / "metrics_by_fold.csv", index=False)
    summary = {"n_folds": len(frame), "feature_mode": feature_mode}
    for column in frame.columns:
        if column != "fold":
            summary[f"{column}_mean"] = float(frame[column].mean())
            summary[f"{column}_sample_sd"] = float(frame[column].std(ddof=1))
    summary["dataset"] = config["dataset"]
    summary["selection_metric"] = "validation_aupr"
    summary["threshold"] = config["threshold"]
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return frame, summary
