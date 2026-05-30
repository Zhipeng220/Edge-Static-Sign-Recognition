import pytest
import torch

from gesture_experiment.config import ExperimentConfig
from gesture_experiment.models import build_model


@pytest.mark.parametrize(
    "model_type,use_positional_encoding,use_attention,use_residual,use_anatomical_loss",
    [
        ("mlp", False, False, False, False),
        ("pointnet", False, False, False, False),
        ("gcn", False, False, False, False),
        ("transformer", True, True, False, False),
        ("attn", True, True, True, True),
    ],
)
def test_model_forward_smoke(
    model_type,
    use_positional_encoding,
    use_attention,
    use_residual,
    use_anatomical_loss,
):
    cfg = ExperimentConfig(
        name=f"test_{model_type}",
        model_type=model_type,
        use_positional_encoding=use_positional_encoding,
        use_attention=use_attention,
        use_residual=use_residual,
        use_anatomical_loss=use_anatomical_loss,
        embed_dim=16,
        num_heads=4,
    )
    model = build_model(cfg, num_classes=6).eval()
    x = torch.randn(2, cfg.num_landmarks, cfg.coord_dim)

    with torch.no_grad():
        logits, _, reconstructed = model(x)

    assert logits.shape == (2, 6)
    if use_anatomical_loss:
        assert reconstructed.shape == x.shape
    else:
        assert reconstructed is None
