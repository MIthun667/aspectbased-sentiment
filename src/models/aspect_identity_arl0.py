from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel


class AspectIdentityARL0(nn.Module):
    """
    ARL-0

    Learns an aspect-identity embedding from the contextualized
    target representation produced by pair-RoBERTa.

    Initial experiment:
        - RoBERTa backbone frozen
        - only projection head trained
        - no sentiment supervision
        - no relation supervision
    """

    def __init__(
        self,
        backbone_name: str = "roberta-base",
        projection_dim: int = 256,
        dropout: float = 0.1,
        freeze_backbone: bool = True,
    ):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(
            backbone_name
        )

        hidden_size = self.encoder.config.hidden_size

        if freeze_backbone:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False

        self.projector = nn.Sequential(
            nn.Linear(
                hidden_size,
                hidden_size,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_size),
            nn.Linear(
                hidden_size,
                projection_dim,
            ),
        )

    @staticmethod
    def masked_mean(
        states: torch.Tensor,
        mask: torch.Tensor,
    ):
        weights = (
            mask.unsqueeze(-1)
            .to(states.dtype)
        )

        numerator = (
            states * weights
        ).sum(dim=1)

        denominator = (
            weights.sum(dim=1)
            .clamp_min(1.0)
        )

        return numerator / denominator

    def forward(
        self,
        input_ids,
        attention_mask,
        target_mask,
    ):
        # Frozen backbone does not need gradients.
        backbone_trainable = any(
            p.requires_grad
            for p in self.encoder.parameters()
        )

        if backbone_trainable:
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
        else:
            with torch.no_grad():
                outputs = self.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True,
                )

        hidden = outputs.last_hidden_state

        target_state = self.masked_mean(
            hidden,
            target_mask,
        )

        embedding = self.projector(
            target_state
        )

        embedding = F.normalize(
            embedding,
            p=2,
            dim=-1,
        )

        return embedding
