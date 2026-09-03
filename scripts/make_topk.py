from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from muse_dma.data import read_names


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank candidate drug-microbe pairs from prediction files")
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--drug-names", required=True, type=Path)
    parser.add_argument("--microbe-names", required=True, type=Path)
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    frame = pd.read_csv(args.predictions)
    required = {"drug_idx0", "microbe_idx0", "score"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Prediction file lacks columns: {sorted(required - set(frame.columns))}")
    drugs, microbes = read_names(args.drug_names), read_names(args.microbe_names)
    ranked = frame.sort_values("score", ascending=False).head(args.k).copy()
    ranked.insert(0, "rank", range(1, len(ranked) + 1))
    ranked["drug_name"] = ranked["drug_idx0"].map(dict(enumerate(drugs)))
    ranked["microbe_name"] = ranked["microbe_idx0"].map(dict(enumerate(microbes)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(args.output, index=False, lineterminator="\n")


if __name__ == "__main__":
    main()
