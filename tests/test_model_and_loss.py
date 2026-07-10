import pytest

torch = pytest.importorskip("torch")

from uti_mpc.losses import ProtoMarginLoss
from uti_mpc.models import UTIMPC


def _model():
    return UTIMPC(
        {
            "byte_embedding_dim": 8,
            "branch_channels": 4,
            "byte_dim": 16,
            "time_dim": 16,
            "embedding_dim": 8,
            "se_reduction": 4,
            "attention_heads": 4,
            "windows": [2, 4],
            "ffn_expansion": 2,
            "dropout": 0.0,
            "max_length": 16,
        }
    )


def test_model_output_is_unit_normalized_and_differentiable():
    model = _model()
    byte_tokens = torch.randint(0, 256, (4, 8, 32))
    byte_mask = torch.ones(4, 8, dtype=torch.bool)
    lengths = torch.randn(4, 8)
    length_mask = torch.ones(4, 8, dtype=torch.bool)
    embeddings, details = model(byte_tokens, lengths, byte_mask, length_mask, True)
    assert embeddings.shape == (4, 8)
    assert torch.allclose(embeddings.norm(dim=1), torch.ones(4), atol=1e-5)
    assert details["modality_gate"].shape == (4, 32)
    embeddings.square().mean().backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_protomargin_has_finite_warmup_and_formal_losses():
    embeddings = torch.nn.functional.normalize(torch.randn(8, 16), dim=1).requires_grad_()
    labels = torch.tensor([1, 1, 1, 1, 2, 2, 2, 2])
    criterion = ProtoMarginLoss()
    warmup = criterion(embeddings, labels, "warmup")
    formal = criterion(embeddings, labels, "formal")
    assert torch.isfinite(warmup["total"])
    assert torch.isfinite(formal["total"])
    formal["total"].backward()
    assert embeddings.grad is not None

