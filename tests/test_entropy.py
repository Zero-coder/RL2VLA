import torch

from rl2vla.utils.entropy import EntropyConfig, categorical_entropy_from_logits


def test_entropy_shape() -> None:
    logits = torch.zeros(4, 7, 16)
    entropy = categorical_entropy_from_logits(logits)
    assert entropy.shape == (4, 1)


def test_normalized_uniform_entropy_is_one() -> None:
    logits = torch.zeros(2, 3, 8)
    entropy = categorical_entropy_from_logits(logits, EntropyConfig(aggregation="mean", normalize_by_vocab=True))
    assert torch.allclose(entropy, torch.ones_like(entropy), atol=1e-6)


def test_sum_aggregation() -> None:
    logits = torch.zeros(2, 3, 8)
    entropy = categorical_entropy_from_logits(logits, EntropyConfig(aggregation="sum", normalize_by_vocab=True))
    assert torch.allclose(entropy, torch.full_like(entropy, 3.0), atol=1e-6)
