# Reproducibility protocol

## Fixed evaluation question

For each dataset, every method is evaluated on the same explicit test-positive
and test-negative pairs in each fold. AUPR and AUC use continuous scores. F1,
accuracy, precision, and recall use a fixed threshold of 0.5.

## Leakage control

Static similarity and frozen semantic embeddings are association-independent.
Every association-derived input is reconstructed separately for each fold:

1. read `folds/fold_k/train_pos.csv`;
2. build the drug-microbe training association matrix;
3. derive drug and microbe GIP similarities from that matrix;
4. fuse GIP with static similarity;
5. construct normalized top-20 graphs and association profiles;
6. train on `train.csv`, select the checkpoint by validation AUPR on
   `val.csv`, and evaluate once on `test.csv`.

Validation and test associations are never used to construct graph edges,
association profiles, GIP similarities, or model-selection targets.

## Semantic embeddings

The four biomedical language models are frozen. Type-prefixed entity names are
mean pooled with attention-mask correction and L2-normalized. The four views
are concatenated and standardized independently for drug and microbe entities.
Resolved Hugging Face commit identifiers are stored next to generated arrays.

## Training configuration

- hidden dimension: 320;
- pair MLP: 1280 -> 320 -> 160 -> 1;
- learning rate: `5e-5`;
- dropout: `0.30`;
- weight decay: `5e-5`;
- similarity neighbors: 20;
- propagation depth: 2;
- SNF iterations: 8;
- objective: positive-class-weighted binary cross-entropy with logits;
- checkpoint selection: validation AUPR;
- random seed base: `20260707`.

Dataset-specific values and complete machine-readable settings are in
`configs/`.

## Commands

```bash
python scripts/prepare_data.py --repository-root .
python scripts/audit_data.py --repository-root .
python scripts/audit_splits.py --repository-root .
pytest -q
```

Generate embeddings and run one dataset:

```bash
python scripts/extract_embeddings.py \
  --config configs/imdad.json \
  --dataset-dir data/IMDAD \
  --models pubmedbert biobert biolinkbert biomedlm

python main.py \
  --config configs/imdad.json \
  --data-root data/IMDAD \
  --output-dir runs/imdad
```

The same interface applies to aBiofilm and MDAD-2470.
