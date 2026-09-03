import torch

from muse_dma.model import MUSEDMA


def test_model_shapes():
    model = MUSEDMA(drug_dim=12, microbe_dim=10, hidden=32, dropout=0.0)
    drug = torch.randn(7, 12)
    microbe = torch.randn(5, 10)
    drug_adj = torch.eye(7)
    microbe_adj = torch.eye(5)
    pairs = torch.tensor([[0, 0], [2, 3], [6, 4]])
    scores = model(drug, microbe, drug_adj, microbe_adj, pairs)
    assert scores.shape == (3,)
