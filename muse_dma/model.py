from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class MUSEDMA(nn.Module):
    """MUSE-DMA semantic-structural association predictor."""

    def __init__(self, drug_dim: int, microbe_dim: int, hidden: int = 320, dropout: float = 0.30):
        super().__init__()
        if hidden % 2:
            raise ValueError("hidden must be even for the bidirectional LSTM")
        self.drug_projection = nn.Linear(drug_dim, hidden)
        self.microbe_projection = nn.Linear(microbe_dim, hidden)
        self.sequence_encoder = nn.LSTM(
            hidden, hidden // 2, bidirectional=True, batch_first=True
        )
        self.depth_attention = nn.Sequential(
            nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1)
        )
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden)
        self.pair_scorer = nn.Sequential(
            nn.Linear(hidden * 4, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def _encode_type(self, features, adjacency, projection):
        local = F.relu(projection(features))
        one_hop = adjacency @ local
        two_hop = adjacency @ one_hop
        propagation_sequence = torch.stack([local, one_hop, two_hop], dim=1)
        encoded, _ = self.sequence_encoder(propagation_sequence)
        weights = torch.softmax(self.depth_attention(encoded), dim=1)
        aggregate = (weights * encoded).sum(dim=1)
        return self.layer_norm(self.dropout(aggregate) + local)

    def encode_entities(self, drug_features, microbe_features, drug_adjacency, microbe_adjacency):
        drug = self._encode_type(drug_features, drug_adjacency, self.drug_projection)
        microbe = self._encode_type(microbe_features, microbe_adjacency, self.microbe_projection)
        return drug, microbe

    @staticmethod
    def pair_representation(drug_embedding, microbe_embedding):
        return torch.cat(
            [
                drug_embedding,
                microbe_embedding,
                drug_embedding * microbe_embedding,
                torch.abs(drug_embedding - microbe_embedding),
            ],
            dim=1,
        )

    def score_pairs(self, drug_embeddings, microbe_embeddings, pair_indices):
        drug = drug_embeddings[pair_indices[:, 0]]
        microbe = microbe_embeddings[pair_indices[:, 1]]
        return self.pair_scorer(self.pair_representation(drug, microbe)).squeeze(1)

    def forward(self, drug_features, microbe_features, drug_adjacency, microbe_adjacency, pair_indices):
        drug, microbe = self.encode_entities(
            drug_features, microbe_features, drug_adjacency, microbe_adjacency
        )
        return self.score_pairs(drug, microbe, pair_indices)
