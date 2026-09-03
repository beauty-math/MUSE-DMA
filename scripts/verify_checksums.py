from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the release SHA-256 manifest")
    parser.add_argument("--repository-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.repository_root.resolve()
    manifest = root / "SHA256SUMS"
    checked = 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        path = root / Path(relative)
        if not path.is_file():
            raise RuntimeError(f"Manifest file is missing: {relative}")
        observed = sha256(path)
        if observed != expected:
            raise RuntimeError(f"Checksum mismatch: {relative}")
        checked += 1
    print(f"PASS: verified {checked} files")


if __name__ == "__main__":
    main()
