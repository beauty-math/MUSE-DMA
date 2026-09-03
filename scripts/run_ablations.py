from __future__ import annotations

import argparse
from pathlib import Path

from muse_dma.data import load_config
from muse_dma.training import run_cross_validation


def main() -> None:
    parser = argparse.ArgumentParser(description="Run predefined MUSE-DMA ablations")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["full", "no_text", "no_graph", "text_only"],
        choices=["full", "no_text", "no_graph", "text_only"],
    )
    args = parser.parse_args()
    config = load_config(args.config)
    for mode in args.modes:
        run_cross_validation(config, args.data_root, args.output_root / mode, feature_mode=mode)


if __name__ == "__main__":
    main()
