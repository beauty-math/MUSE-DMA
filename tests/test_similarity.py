import numpy as np

from muse_dma.similarity import gaussian_interaction_profile, topk_normalized_adjacency


def test_similarity_outputs_are_finite_and_symmetric():
    profiles = np.asarray([[1, 0, 1], [0, 1, 0], [1, 1, 0]], dtype=np.float32)
    similarity = gaussian_interaction_profile(profiles)
    adjacency = topk_normalized_adjacency(similarity, k=2)
    assert similarity.shape == (3, 3)
    assert adjacency.shape == (3, 3)
    assert np.isfinite(similarity).all()
    assert np.isfinite(adjacency).all()
    assert np.allclose(similarity, similarity.T)
    assert np.allclose(adjacency, adjacency.T)
