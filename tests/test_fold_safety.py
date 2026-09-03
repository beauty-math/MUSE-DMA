import pandas as pd

from muse_dma.features import training_association_matrix


def test_training_matrix_uses_only_supplied_positive_pairs():
    train_positive = pd.DataFrame({
        "drug_idx0": [0, 2], "microbe_idx0": [1, 0], "label": [1, 1]
    })
    matrix = training_association_matrix(train_positive, n_drugs=4, n_microbes=3)
    assert matrix.sum() == 2
    assert matrix[0, 1] == 1
    assert matrix[2, 0] == 1
    assert matrix[3, 2] == 0
