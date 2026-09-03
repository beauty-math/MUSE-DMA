from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from prepare_data import DATASETS, read_names


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit bundled benchmark data")
    parser.add_argument("--repository-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.repository_root.resolve()
    report = {"status": "PASS", "datasets": {}}
    for key, specification in DATASETS.items():
        dataset = root / "data" / key
        required = [
            dataset / specification["drug_names"],
            dataset / specification["microbe_names"],
            dataset / specification["associations"],
            dataset / "processed" / "positive_associations.csv",
        ]
        for path in required:
            if not path.is_file() or path.stat().st_size == 0:
                raise RuntimeError(f"Missing or empty data file: {path}")
        drugs = read_names(required[0])
        microbes = read_names(required[1])
        associations = pd.read_csv(required[3])
        observed = (len(drugs), len(microbes), len(associations))
        if observed != specification["expected"]:
            raise RuntimeError(f"{key}: observed {observed}, expected {specification['expected']}")
        if associations.duplicated(["drug_idx0", "microbe_idx0"]).any():
            raise RuntimeError(f"{key}: duplicate canonical associations")
        if not associations["label"].eq(1).all():
            raise RuntimeError(f"{key}: canonical positives contain non-positive labels")
        report["datasets"][key] = {
            "n_drugs": observed[0],
            "n_microbes": observed[1],
            "n_positive_associations": observed[2],
            "raw_association_sha256": sha256(required[2]),
            "canonical_association_sha256": sha256(required[3]),
        }
    output = root / "data" / "data_audit.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
