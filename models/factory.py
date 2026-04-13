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
            hyper_hidden_dim=model_cfg.get("hyper_hidden_dim"),
            condition_kan1=bool(model_cfg.get("hyperkan_condition_kan1", True)),
            mixture_temperature=float(model_cfg.get("mixture_temperature", 1.0)),
            site_op_factorized=bool(model_cfg.get("site_op_factorized", False)),
            num_sites=model_cfg.get("num_sites"),
            num_ops=model_cfg.get("num_ops"),
            action_to_site_idx=model_cfg.get("action_to_site_idx"),
            action_to_op_idx=model_cfg.get("action_to_op_idx"),
            use_frontier_head=bool(model_cfg.get("use_frontier_head", False)),
        )
    raise ValueError(f"Unknown model type: {model_type}")
