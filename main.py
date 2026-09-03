from __future__ import annotations

import argparse
from pathlib import Path

from muse_dma.data import load_config
from muse_dma.training import run_cross_validation


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MUSE-DMA cross-validation")
    parser.add_argument("--config", required=True, help="Dataset configuration JSON")
    parser.add_argument("--data-root", required=True, help="Dataset directory under data/")
    parser.add_argument("--output-dir", required=True, help="New run directory")
    parser.add_argument(
        "--feature-mode",
        default="full",
        choices=["full", "no_text", "no_graph", "text_only"],
        help="Full model or a predefined ablation",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    run_cross_validation(
        config,
        Path(args.data_root),
        Path(args.output_dir),
        feature_mode=args.feature_mode,
    )


if __name__ == "__main__":
    main()
