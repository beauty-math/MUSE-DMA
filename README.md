# MUSE-DMA

MUSE-DMA is a reproducible implementation of **Multi-encoder Unified
Semantic-structural Embedding for Drug-Microbe Association prediction**.

The repository contains the model code, the three benchmark datasets used by
the study, the exact five-fold pair files used for evaluation, and audit
utilities.

## Model

1. Drug and microbe names are encoded offline with PubMedBERT, BioBERT,
   BioLinkBERT, and BioMedLM.
2. Standardized semantic views are concatenated with fold-safe structural
   context: static similarity, train-fold GIP similarity, and train-fold
   association profiles.
3. A normalized top-20 graph produces local, one-hop, and two-hop entity
   states. A shared bidirectional LSTM and attention layer aggregate them.
4. Candidate pairs are represented as
   `[h_d, h_m, h_d * h_m, abs(h_d - h_m)]` and scored by an MLP.

Validation AUPR selects checkpoints. Test pairs never contribute to graph or
association-derived feature construction.

## Repository layout

```text
MUSE-DMA/
|-- configs/                 dataset and training configurations
|-- data/                    real raw data, canonical tables, frozen folds
|-- docs/                    provenance and reproducibility notes
|-- muse_dma/                model, features, metrics, and training code
|-- scripts/                 data preparation, embedding, split, and audits
|-- tests/                   unit and leakage-control tests
|-- main.py                  cross-validation entry point
|-- requirements.txt
`-- SHA256SUMS
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Verify the release

```bash
python scripts/audit_data.py --repository-root .
python scripts/audit_splits.py --repository-root .
python scripts/verify_checksums.py --repository-root .
pytest -q
```

Both audits must finish without errors before training.

## Create semantic embeddings

The embeddings are derived artifacts and are not duplicated in Git. Generate
them from the pinned model identifiers:

```bash
python scripts/extract_embeddings.py \
  --config configs/imdad.json \
  --dataset-dir data/IMDAD \
  --models pubmedbert biobert biolinkbert biomedlm
```

Repeat with `configs/abiofilm.json` and `configs/mdad2470.json`.

## Run five-fold evaluation

```bash
python main.py \
  --config configs/imdad.json \
  --data-root data/IMDAD \
  --output-dir runs/imdad
```

For a predefined ablation, add one of:

```text
--feature-mode no_text
--feature-mode no_graph
--feature-mode text_only
```

See [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for the complete protocol and
[DATA_PROVENANCE.md](docs/DATA_PROVENANCE.md) for source commits and data terms.
