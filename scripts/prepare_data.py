from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


DATASETS = {
    "IMDAD": {
        "drug_names": "raw/mdd/drug/drug_name.txt",
        "microbe_names": "raw/mdd/microbe/microbe_name.txt",
        "associations": "raw/mdd/adj/microbe_drug_adj.txt",
        "association_columns": 2,
        "expected": (1209, 172, 2268),
    },
    "aBiofilm": {
        "drug_names": "raw/drugs.txt",
        "microbe_names": "raw/microbes.txt",
        "associations": "raw/adj.txt",
        "association_columns": 3,
        "expected": (1720, 140, 2884),
    },
    "MDAD2470": {
        "drug_names": "raw/drugs.txt",
        "microbe_names": "raw/microbes.txt",
        "associations": "raw/adj.txt",
        "association_columns": 3,
        "expected": (1373, 173, 2470),
    },
}


def read_names(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def prepare(dataset_root: Path, key: str) -> dict:
    specification = DATASETS[key]
    drugs = read_names(dataset_root / specification["drug_names"])
    microbes = read_names(dataset_root / specification["microbe_names"])
    raw = pd.read_csv(
        dataset_root / specification["associations"],
        sep=r"\s+",
        header=None,
        encoding="utf-8-sig",
    )
    if raw.shape[1] != specification["association_columns"]:
        raise ValueError(f"Unexpected association shape for {key}: {raw.shape}")
    raw = raw.iloc[:, :2].astype(int)
    raw.columns = ["drug_id1", "microbe_id1"]
    raw = raw.drop_duplicates().reset_index(drop=True)
    raw["drug_idx0"] = raw["drug_id1"] - 1
    raw["microbe_idx0"] = raw["microbe_id1"] - 1
    raw["drug_name"] = raw["drug_idx0"].map(dict(enumerate(drugs)))
    raw["microbe_name"] = raw["microbe_idx0"].map(dict(enumerate(microbes)))
    raw["label"] = 1
    if raw[["drug_name", "microbe_name"]].isna().any().any():
        raise ValueError(f"Association IDs are out of range for {key}")

    expected_drugs, expected_microbes, expected_pairs = specification["expected"]
    observed = (len(drugs), len(microbes), len(raw))
    if observed != specification["expected"]:
        raise ValueError(f"Unexpected {key} dimensions: {observed}, expected {specification['expected']}")

    processed = dataset_root / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    (processed / "drug_names.txt").write_text("\n".join(drugs) + "\n", encoding="utf-8")
    (processed / "microbe_names.txt").write_text("\n".join(microbes) + "\n", encoding="utf-8")
    columns = [
        "drug_idx0",
        "microbe_idx0",
        "drug_id1",
        "microbe_id1",
        "drug_name",
        "microbe_name",
        "label",
    ]
    raw[columns].to_csv(processed / "positive_associations.csv", index=False, lineterminator="\n")
    manifest = {
        "dataset": "MDAD-2470" if key == "MDAD2470" else key,
        "n_drugs": expected_drugs,
        "n_microbes": expected_microbes,
        "n_positive_associations": expected_pairs,
        "density": expected_pairs / (expected_drugs * expected_microbes),
        "id_convention": "zero-based indices in canonical CSV; one-based IDs retained for traceability",
    }
    (dataset_root / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Create canonical benchmark tables from bundled raw data")
    parser.add_argument("--repository-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.repository_root.resolve()
    outputs = {}
    for key in DATASETS:
        outputs[key] = prepare(root / "data" / key, key)
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
