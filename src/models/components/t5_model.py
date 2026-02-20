from __future__ import annotations
"""Wrapper for Hugging Face's T5ForConditionalGeneration, integrating for PyTorch-Lightning and Hydra."""

from typing import Any, Dict, Optional

import torch
from torch import nn
from transformers import T5Config, T5ForConditionalGeneration


class T5Model(nn.Module):
    """T5 model wrapper."""

    def __init__(
        self,
        pretrained_model_name_or_path: str | None = "t5-base",
        config_kwargs: Optional[Dict[str, Any]] = None,
        enable_gradient_checkpointing: bool = True,
        dropout_rate: float | None = None,
    ) -> None:
        """Create a :class:`T5Model`.

        Parameters
        ----------
        pretrained_model_name_or_path
            Name of, or path to, a Hugging Face checkpoint.  If *None*, a new
            model is initialised from scratch using *config_kwargs*.
        config_kwargs
            Additional keyword arguments forwarded to
            :class:`~transformers.T5Config` when *pretrained_model_name_or_path*
            is *None*.
        enable_gradient_checkpointing
            Whether to activate gradient checkpointing for memory efficiency
            during training.
        """
        super().__init__()

        if pretrained_model_name_or_path is None:
            if config_kwargs is None:
                config_kwargs = {}
            # Allow overriding dropout via config when training from scratch
            if dropout_rate is not None:
                config_kwargs = {**config_kwargs, "dropout_rate": dropout_rate}
            config = T5Config(**config_kwargs)
            self.model = T5ForConditionalGeneration(config)
        else:
            # Load base config and optionally override selected fields (e.g., dropout)
            try:
                base_cfg = T5Config.from_pretrained(pretrained_model_name_or_path)
            except Exception:
                base_cfg = T5Config()
            if config_kwargs:
                for k, v in config_kwargs.items():
                    setattr(base_cfg, k, v)
            if dropout_rate is not None:
                base_cfg.dropout_rate = dropout_rate

            self.model = T5ForConditionalGeneration.from_pretrained(
                pretrained_model_name_or_path,
                config=base_cfg,
            )

        if enable_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

    def forward(self, **kwargs):
        return self.model(**kwargs)

    @property
    def config(self) -> T5Config:  # type: ignore[override]
        return self.model.config

    def resize_token_embeddings(self, new_num_tokens: int) -> None:
        self.model.resize_token_embeddings(new_num_tokens)

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs) 
