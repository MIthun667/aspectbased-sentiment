from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel


class TargetEvidenceTRRL2(nn.Module):
    """
    TRRL-2

    Learn a target-context sentiment relation representation using
    TARGET-CONDITIONED EVIDENCE rather than global CLS.

    Pipeline:

        pair-RoBERTa
            ↓
        target representation h_t
        sentence token states H
            ↓
        target-conditioned evidence attention
            ↓
        evidence representation h_e
            ↓
        relation features:
            [h_t, h_e, h_t * h_e, |h_t - h_e|]
            ↓
        relation projector
            ↓
        normalized relation embedding

    Initial experiment:
        - RoBERTa frozen
        - evidence scorer + relation projector trainable
        - no sentiment classifier
        - no graph / syntax
    """

    def __init__(
        self,
        backbone_name: str = "roberta-base",
        relation_dim: int = 256,
        attention_dim: int = 256,
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

        # Target-conditioned evidence scorer.
        self.target_query = nn.Linear(
            hidden_size,
            attention_dim,
            bias=False,
        )

        self.token_key = nn.Linear(
            hidden_size,
            attention_dim,
            bias=False,
        )

        self.token_value = nn.Linear(
            hidden_size,
            hidden_size,
            bias=False,
        )

        self.attention_dim = attention_dim

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
    ) -> torch.Tensor:

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
        sentence_mask,
        target_mask,
        return_attention: bool = False,
    ):
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

        # Second-sequence target representation.
        target_state = self.masked_mean(
            hidden,
            target_mask,
        )

        # Sentence sequence only.
        q = self.target_query(
            target_state
        ).unsqueeze(1)

        k = self.token_key(
            hidden
        )

        scores = (
            q * k
        ).sum(dim=-1)

        scores = (
            scores
            / math.sqrt(
                self.attention_dim
            )
        )

        # Only original sentence tokens may act as evidence.
        evidence_mask = sentence_mask.bool()

        scores = scores.masked_fill(
            ~evidence_mask,
            torch.finfo(
                scores.dtype
            ).min,
        )

        attention = F.softmax(
            scores,
            dim=-1,
        )

        values = self.token_value(
            hidden
        )

        evidence_state = torch.bmm(
            attention.unsqueeze(1),
            values,
        ).squeeze(1)

        interaction = (
            target_state
            * evidence_state
        )

        difference = torch.abs(
            target_state
            - evidence_state
        )

        relation_input = torch.cat(
            [
                target_state,
                evidence_state,
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

        if return_attention:
            return (
                relation_embedding,
                attention,
            )

        return relation_embedding
