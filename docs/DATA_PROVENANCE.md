# Data provenance

## IMDAD

- Upstream repository: `https://github.com/pingxuan-hlju/pcmda`
- Retrieved commit: `71ab13a4c619afa1c6287327760766b85138f72b`
- Upstream archive: `mdd.rar`
- Release location: `data/IMDAD/raw/mdd/`
- Dimensions verified from the bundled names and association edge list:
  1,209 drugs, 172 microbes, and 2,268 positive associations.

The misspelled upstream filename `drug_fringer_simi.txt` is retained unchanged
to preserve byte-level provenance.

## aBiofilm and MDAD-2470

- Upstream repository: `https://github.com/Yue-Yuu/SCSMDA-master`
- Retrieved commit: `5562cb6cf629993c5b53b000fe0a53ca30e66456`
- Upstream directories: `data/aBiofilm/` and `data/MDAD/`
- Release locations: `data/aBiofilm/raw/` and `data/MDAD2470/raw/`
- Verified dimensions:
  - aBiofilm: 1,720 drugs, 140 microbes, 2,884 positive associations;
  - MDAD-2470: 1,373 drugs, 173 microbes, 2,470 positive associations.

## Canonicalization

`scripts/prepare_data.py` reads the original name and edge-list files without
reordering entities. It creates a canonical positive-association CSV with both
zero-based indices and the original one-based IDs. It does not filter or add
associations.

## Frozen folds

All datasets use seed `20260707` and five folds. Positive and sampled-negative
pairs are explicit files rather than being regenerated inside individual model
implementations. The supplied files were regenerated from the original split
algorithms and matched the frozen experiment manifest for all 90 formally
tracked files by SHA-256 before release assembly.

Run `python scripts/audit_splits.py --repository-root .` to reproduce the
pair-disjointness, label, role-composition, and known-positive checks and to
write a complete current checksum manifest.
