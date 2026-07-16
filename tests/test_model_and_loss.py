import pytest

torch = pytest.importorskip("torch")

from uti_mpc.losses import ProtoMarginLoss
from uti_mpc.data.augmentation import make_v3_view
from uti_mpc.losses import InformationGeometryBoundaryLoss
from uti_mpc.models import UTIMPC, UTIMPCV3, predict_v3


def _model():
    return UTIMPC(
        {
            "byte_embedding_dim": 8,
            "branch_channels": 4,
            "byte_dim": 16,
            "time_dim": 16,
            "embedding_dim": 8,
            "bgi_residual_blocks": 1,
            "twt_depth": 2,
            "shifted_windows": True,
            "fusion_residual": True,
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


def test_arcface_auxiliary_loss_is_formal_stage_only_and_differentiable():
    embeddings = torch.nn.functional.normalize(torch.randn(8, 16), dim=1).requires_grad_()
    labels = torch.tensor([1, 1, 1, 1, 4, 4, 4, 4])
    criterion = ProtoMarginLoss(
        known_classes=[1, 4],
        embedding_dim=16,
        lambda_arcface=0.3,
        arcface_scale=16.0,
        arcface_margin=0.2,
    )
    warmup = criterion(embeddings, labels, "warmup")
    assert warmup["arcface"].item() == 0.0
    formal = criterion(embeddings, labels, "formal")
    assert torch.isfinite(formal["arcface"])
    assert formal["arcface"] > 0.0
    formal["total"].backward()
    assert embeddings.grad is not None
    assert criterion.arcface.weight.grad is not None


def test_v2_hierarchical_cross_modal_model_masks_padding_and_backpropagates():
    model = UTIMPC(
        {
            "hierarchical_bgi": True,
            "cross_modal_fusion": True,
            "byte_embedding_dim": 8,
            "byte_dim": 16,
            "time_dim": 16,
            "embedding_dim": 8,
            "byte_attention_heads": 4,
            "byte_packet_layers": 1,
            "cross_modal_dim": 16,
            "cross_attention_heads": 4,
            "attention_heads": 4,
            "windows": [2, 4],
            "ffn_expansion": 2,
            "dropout": 0.0,
            "max_length": 16,
            "max_packets": 16,
            "byte_width": 64,
            "temporal_input_dim": 13,
            "twt_depth": 2,
            "shifted_windows": True,
        }
    )
    byte_tokens = torch.randint(0, 256, (2, 8, 64))
    byte_mask = torch.tensor(
        [[True] * 8, [True, True, False, False, False, False, False, False]]
    )
    altered = byte_tokens.clone()
    altered[1, 2:] = torch.randint(0, 256, altered[1, 2:].shape)
    lengths = torch.randn(2, 4, 13)
    length_mask = torch.tensor([[True] * 4, [True, True, False, False]])
    model.eval()
    first, details = model(byte_tokens, lengths, byte_mask, length_mask, True)
    second = model(altered, lengths, byte_mask, length_mask)
    assert torch.allclose(first[1], second[1], atol=1e-6)
    assert details["modality_gate"].shape == (2, 2)
    assert torch.allclose(first.norm(dim=1), torch.ones(2), atol=1e-5)
    first.sum().backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_subcenter_loss_and_ema_weighting_are_finite_and_trainable():
    embeddings = torch.nn.functional.normalize(torch.randn(12, 16), dim=1).requires_grad_()
    labels = torch.tensor([1] * 6 + [4] * 6)
    criterion = ProtoMarginLoss(
        known_classes=[1, 4],
        embedding_dim=16,
        subcenters_per_class=3,
        lambda_intra=1.0,
        lambda_inter=1.0,
        lambda_diversity=0.2,
        loss_weighting="ema",
    )
    losses = criterion(embeddings, labels, "formal")
    assert torch.isfinite(losses["total"])
    assert "weight_intra" in losses
    losses["total"].backward()
    assert criterion.subcenters.centers.grad is not None


def _v3_model():
    return UTIMPCV3(
        {
            "payload_bytes": 12,
            "max_packets": 4,
            "max_bursts": 3,
            "byte_embedding_dim": 8,
            "packet_dim": 24,
            "packet_heads": 4,
            "packet_layers": 1,
            "burst_dim": 16,
            "burst_heads": 4,
            "burst_layers": 1,
            "embedding_dim": 8,
            "subprototypes_per_class": 2,
            "dropout": 0.0,
        },
        [1, 2],
    )


def test_v3_model_geometry_augmentation_and_formal_loss():
    model = _v3_model()
    batch = {
        "payload_tokens": torch.randint(0, 256, (4, 4, 12)),
        "payload_mask": torch.ones(4, 4, 12, dtype=torch.bool),
        "packet_features": torch.rand(4, 4, 16),
        "packet_mask": torch.ones(4, 4, dtype=torch.bool),
        "burst_features": torch.rand(4, 3, 8),
        "burst_mask": torch.ones(4, 3, dtype=torch.bool),
    }
    first = make_v3_view(batch, packet_drop=0.2)
    second = make_v3_view(batch, packet_drop=0.2)
    input_keys = (
        "payload_tokens",
        "payload_mask",
        "packet_features",
        "packet_mask",
        "burst_features",
        "burst_mask",
    )
    first_embedding, first_details = model(
        **{key: first[key] for key in input_keys}, return_details=True
    )
    second_embedding, second_details = model(
        **{key: second[key] for key in input_keys}, return_details=True
    )
    assert torch.allclose(first_embedding.norm(dim=1), torch.ones(4), atol=1e-5)
    labels = torch.tensor([1, 1, 2, 2])
    model.geometry.initialize_from_embeddings(first_embedding.detach(), labels)
    criterion = InformationGeometryBoundaryLoss(model.geometry, {})
    losses = criterion(
        first_embedding,
        second_embedding,
        labels,
        first_details,
        second_details,
        first["reconstruction_target"],
        second["reconstruction_target"],
        first["reconstruction_mask"],
        second["reconstruction_mask"],
        "formal",
    )
    assert torch.isfinite(losses["total"])
    losses["total"].backward()
    assert model.geometry.prototypes.grad is not None
    prediction = predict_v3(model, first_embedding.detach())
    assert prediction["normalized_scores"].shape == (4,)
    assert torch.all(model.geometry.radii() > 0)
