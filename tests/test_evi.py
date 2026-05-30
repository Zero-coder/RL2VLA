import torch

from rl2vla.models.evi import EVIConfig, EntropyFiLM
from rl2vla.models.value_heads import RL2VLAValueHead, ValueHeadConfig


def test_film_identity_initialization() -> None:
    film = EntropyFiLM(EVIConfig(value_dim=8, hidden_dim=4))
    features = torch.randn(3, 8)
    entropy = torch.rand(3, 1)
    out = film(features, entropy)
    assert torch.allclose(out, features, atol=1e-6)


def test_value_head_outputs_scalar() -> None:
    head = RL2VLAValueHead(ValueHeadConfig(input_dim=16, hidden_dim=8, conditioning="film"))
    features = torch.randn(5, 16)
    entropy = torch.rand(5, 1)
    values = head(features, entropy)
    assert values.shape == (5,)
