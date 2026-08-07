from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel


class TargetRelationTRRL1(nn.Module):
    """
    TRRL-1

    Learns a target-context RELATION embedding.

    The target identity itself is not the object of interest.
    Instead, we model how a contextualized target relates to
    the sentence-level representation.

    Relation input:

        target
        context
        target * context
        |target - context|

    Initial experiment:
        - frozen RoBERTa
        - train relation projector only
        - no sentiment classifier
        - no graph / syntax
    """

    def __init__(
        self,
        backbone_name: str = "roberta-base",
        relation_dim: int = 256,
        dropout: float = 0.1,
        freeze_backbone: bool = True,
    ):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(
            backbone_name
        )

        hidden_size = self.encoder.config.hidden_size

        if freeze_backbone:
            for p in self.encoder.parameters():
                p.requires_grad = False

        relation_input_dim = hidden_size * 4

        self.relation_projector = nn.Sequential(
            nn.LayerNorm(
                relation_input_dim
            ),
            nn.Linear(
                relation_input_dim,
                hidden_size,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                hidden_size,
                relation_dim,
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
        trainable_backbone = any(
            p.requires_grad
            for p in self.encoder.parameters()
        )

        if trainable_backbone:
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

        context_state = hidden[:, 0]

        target_state = self.masked_mean(
            hidden,
            target_mask,
        )

        interaction = (
            target_state
            * context_state
        )

        difference = torch.abs(
            target_state
            - context_state
        )

        relation_input = torch.cat(
            [
                target_state,
                context_state,
                interaction,
                difference,
            ],
            dim=-1,
        )

        relation_embedding = (
            self.relation_projector(
                relation_input
            )
        )

        relation_embedding = F.normalize(
            relation_embedding,
            p=2,
            dim=-1,
        )

        return relation_embedding
