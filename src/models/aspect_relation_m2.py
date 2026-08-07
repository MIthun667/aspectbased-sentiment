from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel
from transformers.modeling_outputs import SequenceClassifierOutput


class AspectEvidenceRelationEncoder(nn.Module):
    """
    Build a relation representation r_i = R(h_target, h_i)
    for every sentence token h_i.

    The relation is factorized across K heads:

        t_k = W_t^k(h_target)
        e_k = W_e^k(h_i)

        r_i^k = t_k * e_k

    The K relation factors are concatenated and projected back
    into hidden_size.

    The resulting relation tokens are then processed by one
    Transformer encoder layer.
    """

    def __init__(
        self,
        hidden_size: int,
        relation_dim: int = 128,
        num_relation_heads: int = 4,
        transformer_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.relation_dim = relation_dim
        self.num_relation_heads = num_relation_heads

        self.target_projections = nn.ModuleList(
            [
                nn.Linear(hidden_size, relation_dim)
                for _ in range(num_relation_heads)
            ]
        )

        self.evidence_projections = nn.ModuleList(
            [
                nn.Linear(hidden_size, relation_dim)
                for _ in range(num_relation_heads)
            ]
        )

        relation_input_dim = (
            relation_dim * num_relation_heads
        )

        self.relation_projection = nn.Sequential(
            nn.Linear(
                relation_input_dim,
                hidden_size,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_size),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=transformer_heads,
            dim_feedforward=hidden_size * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.relation_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=1,
        )

    def forward(
        self,
        sentence_states: torch.Tensor,
        target_state: torch.Tensor,
        sentence_mask: torch.Tensor,
    ):
        relation_parts = []

        for target_proj, evidence_proj in zip(
            self.target_projections,
            self.evidence_projections,
        ):
            target_factor = target_proj(
                target_state
            ).unsqueeze(1)

            evidence_factor = evidence_proj(
                sentence_states
            )

            relation_part = (
                target_factor
                * evidence_factor
            )

            relation_parts.append(
                relation_part
            )

        relation_tokens = torch.cat(
            relation_parts,
            dim=-1,
        )

        relation_tokens = self.relation_projection(
            relation_tokens
        )

        padding_mask = ~sentence_mask.bool()

        relation_tokens = self.relation_transformer(
            relation_tokens,
            src_key_padding_mask=padding_mask,
        )

        relation_tokens = (
            relation_tokens
            * sentence_mask.unsqueeze(-1).to(
                relation_tokens.dtype
            )
        )

        return relation_tokens


class AspectRelationM2(nn.Module):
    """
    M2:
        Pair-RoBERTa
          ->
        aspect-token relation objects
          ->
        relation-level Transformer
          ->
        relation pooling
          ->
        CLS residual classification
    """

    def __init__(
        self,
        backbone_name: str = "roberta-base",
        num_labels: int = 3,
        relation_dim: int = 128,
        num_relation_heads: int = 4,
        transformer_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(
            backbone_name
        )

        hidden_size = self.encoder.config.hidden_size

        self.relation_encoder = (
            AspectEvidenceRelationEncoder(
                hidden_size=hidden_size,
                relation_dim=relation_dim,
                num_relation_heads=num_relation_heads,
                transformer_heads=transformer_heads,
                dropout=dropout,
            )
        )

        self.pool_relation = nn.Linear(
            hidden_size,
            hidden_size,
            bias=False,
        )

        self.pool_target = nn.Linear(
            hidden_size,
            hidden_size,
            bias=False,
        )

        self.final_norm = nn.LayerNorm(
            hidden_size
        )

        self.relation_to_hidden = nn.Linear(
            hidden_size,
            hidden_size,
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(
                hidden_size,
                num_labels,
            ),
        )

    @staticmethod
    def masked_mean(
        states: torch.Tensor,
        mask: torch.Tensor,
    ):
        weights = mask.unsqueeze(-1).to(
            states.dtype
        )

        numerator = (
            states * weights
        ).sum(dim=1)

        denominator = weights.sum(
            dim=1
        ).clamp_min(1.0)

        return numerator / denominator

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        sentence_mask=None,
        target_mask=None,
        labels=None,
        **kwargs,
    ):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        hidden = outputs.last_hidden_state

        cls_state = hidden[:, 0]

        target_state = self.masked_mean(
            hidden,
            target_mask,
        )

        relation_tokens = self.relation_encoder(
            sentence_states=hidden,
            target_state=target_state,
            sentence_mask=sentence_mask,
        )

        relation_query = self.pool_relation(
            relation_tokens
        )

        target_query = self.pool_target(
            target_state
        ).unsqueeze(1)

        pool_scores = (
            relation_query
            * target_query
        ).sum(dim=-1)

        pool_scores = pool_scores / math.sqrt(
            hidden.size(-1)
        )

        pool_scores = pool_scores.masked_fill(
            ~sentence_mask.bool(),
            torch.finfo(
                pool_scores.dtype
            ).min,
        )

        relation_attention = F.softmax(
            pool_scores,
            dim=-1,
        )

        relation_state = torch.bmm(
            relation_attention.unsqueeze(1),
            relation_tokens,
        ).squeeze(1)

        relation_state = (
            self.relation_to_hidden(
                relation_state
            )
        )

        final_state = self.final_norm(
            cls_state + relation_state
        )

        logits = self.classifier(
            final_state
        )

        loss = None

        if labels is not None:
            loss = F.cross_entropy(
                logits,
                labels,
            )

        result = SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )

        result.relation_tokens = (
            relation_tokens
        )

        result.relation_attention = (
            relation_attention
        )

        return result
