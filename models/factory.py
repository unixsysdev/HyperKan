from __future__ import annotations

from typing import Any

from models.hyperkan_head import HyperKANPolicy
from models.kan_head import StaticKANPolicy
from models.mlp_policy import MLPPolicy


def create_model(model_type: str, config: dict[str, Any]):
    model_cfg = config["model"]
    common_kwargs = {
        "vocab_size": model_cfg["vocab_size"],
        "pad_id": model_cfg["pad_id"],
        "num_actions": model_cfg["num_actions"],
        "embed_dim": model_cfg["embed_dim"],
        "hidden_dim": model_cfg["hidden_dim"],
        "encoder_layers": model_cfg["encoder_layers"],
        "dropout": model_cfg["dropout"],
    }

    if model_type == "mlp":
        return MLPPolicy(**common_kwargs)
    if model_type == "static_kan":
        return StaticKANPolicy(
            **common_kwargs,
            kan_hidden_dim=model_cfg["kan_hidden_dim"],
            basis_dim=model_cfg["spline_basis_dim"],
            num_templates=model_cfg["spline_templates"],
        )
    if model_type == "hyperkan":
        return HyperKANPolicy(
            **common_kwargs,
            kan_hidden_dim=model_cfg["kan_hidden_dim"],
            basis_dim=model_cfg["spline_basis_dim"],
            num_templates=model_cfg["spline_templates"],
        )
    raise ValueError(f"Unknown model type: {model_type}")

