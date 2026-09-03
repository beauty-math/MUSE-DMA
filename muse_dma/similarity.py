from __future__ import annotations

import numpy as np


def normalize_similarity(matrix: np.ndarray) -> np.ndarray:
    matrix = np.nan_to_num(matrix.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    matrix = (matrix + matrix.T) / 2.0
    minimum, maximum = float(matrix.min()), float(matrix.max())
    if maximum > minimum:
        matrix = (matrix - minimum) / (maximum - minimum)
    np.fill_diagonal(matrix, 1.0)
    return matrix.astype(np.float32)


def gaussian_interaction_profile(profiles: np.ndarray) -> np.ndarray:
    profiles = profiles.astype(np.float32)
    squared = (profiles * profiles).sum(axis=1)
    gamma = 1.0 / (squared.mean() + 1e-8)
    distances = squared[:, None] + squared[None, :] - 2.0 * (profiles @ profiles.T)
    return normalize_similarity(np.exp(-gamma * np.maximum(distances, 0.0)))


def topk_normalized_adjacency(similarity: np.ndarray, k: int = 20) -> np.ndarray:
    n = similarity.shape[0]
    adjacency = np.zeros_like(similarity, dtype=np.float32)
    for row in range(n):
        indices = np.argsort(similarity[row])[-(k + 1):]
        indices = indices[indices != row][-k:]
        adjacency[row, indices] = similarity[row, indices]
    adjacency = np.maximum(adjacency, adjacency.T)
    np.fill_diagonal(adjacency, 1.0)
    degree_scale = 1.0 / np.sqrt(adjacency.sum(axis=1) + 1e-8)
    return (degree_scale[:, None] * adjacency * degree_scale[None, :]).astype(np.float32)


def similarity_network_fusion(
    first: np.ndarray,
    second: np.ndarray,
    k: int = 20,
    steps: int = 8,
) -> np.ndarray:
    first, second = normalize_similarity(first), normalize_similarity(second)

    def transition(matrix: np.ndarray) -> np.ndarray:
        output = matrix / (2.0 * ((matrix.sum(1) - np.diag(matrix))[:, None] + 1e-8))
        np.fill_diagonal(output, 0.5)
        return output.astype(np.float32)

    def local_kernel(matrix: np.ndarray) -> np.ndarray:
        output = np.zeros_like(matrix, dtype=np.float32)
        for row in range(matrix.shape[0]):
            indices = np.argsort(matrix[row])[-(k + 1):]
            indices = indices[indices != row][-k:]
            output[row, indices] = matrix[row, indices] / (matrix[row, indices].sum() + 1e-8)
        return output

    probabilities = [transition(first), transition(second)]
    kernels = [local_kernel(first), local_kernel(second)]
    for _ in range(steps):
        probabilities = [
            transition(normalize_similarity(kernels[0] @ probabilities[1] @ kernels[0].T)),
            transition(normalize_similarity(kernels[1] @ probabilities[0] @ kernels[1].T)),
        ]
    return normalize_similarity((probabilities[0] + probabilities[1]) / 2.0)
