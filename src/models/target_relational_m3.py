from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel
from transformers.modeling_outputs import SequenceClassifierOutput


@dataclass
class RelationalSequenceClassifierOutput(SequenceClassifierOutput):
    contrast_embedding: torch.FloatTensor | None = None



class TargetConditionedRelationLayer(nn.Module):
    """
    Same core relational mechanism as M1.

    q_i = W_q(h_i)
    k_j = W_k(h_j)
    v_j = W_v(h_j)
    t   = W_t(h_target)

    target-conditioned query:
        q'_i = q_i * t

    relation:
        A_ij = softmax(q'_i k_j^T / sqrt(d))
    """

    def __init__(
        self,
        hidden_size: int,
        relation_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.relation_dim = relation_dim

        self.query = nn.Linear(
            hidden_size,
            relation_dim,
        )

        self.key = nn.Linear(
            hidden_size,
            relation_dim,
        )

        self.value = nn.Linear(
            hidden_size,
            relation_dim,
        )

        self.target = nn.Linear(
            hidden_size,
            relation_dim,
        )

        self.output = nn.Linear(
            relation_dim,
            hidden_size,
        )

        self.gate = nn.Sequential(
            nn.Linear(
                hidden_size * 2,
                hidden_size,
            ),
            nn.Sigmoid(),
        )

        self.dropout = nn.Dropout(
            dropout
        )

        self.layer_norm = nn.LayerNorm(
            hidden_size
        )

    def forward(
        self,
        sentence_states,
        target_state,
        sentence_mask,
    ):
        q = self.query(
            sentence_states
        )

        k = self.key(
            sentence_states
        )

        v = self.value(
            sentence_states
        )

        target = self.target(
            target_state
        ).unsqueeze(1)

        q = q * target

        scores = torch.matmul(
            q,
            k.transpose(-1, -2),
        )

        scores = (
            scores
            / math.sqrt(
                self.relation_dim
            )
        )

        key_mask = (
            sentence_mask
            .unsqueeze(1)
            .bool()
        )

        scores = scores.masked_fill(
            ~key_mask,
            torch.finfo(
                scores.dtype
            ).min,
        )

        attention = F.softmax(
            scores,
            dim=-1,
        )

        # Zero invalid query rows.
        query_mask = (
            sentence_mask
            .unsqueeze(-1)
            .to(attention.dtype)
        )

        attention = (
            attention
            * sentence_mask
            .unsqueeze(1)
            .to(attention.dtype)
        )

        attention = (
            attention
            / attention.sum(
                dim=-1,
                keepdim=True,
            ).clamp_min(1e-8)
        )

        relational = torch.matmul(
            attention,
            v,
        )

        relational = self.output(
            relational
        )

        relational = self.dropout(
            relational
        )

        target_expanded = (
            target_state
            .unsqueeze(1)
            .expand_as(
                sentence_states
            )
        )

        gate = self.gate(
            torch.cat(
                [
                    sentence_states,
                    target_expanded,
                ],
                dim=-1,
            )
        )

        refined = self.layer_norm(
            sentence_states
            + gate * relational
        )

        refined = (
            refined
            * query_mask
        )

        return (
            refined,
            attention,
            gate,
        )


class TargetRelationalM3(nn.Module):
    """
    M3-A

    Architecture deliberately kept equivalent to M1.

    The only substantive experimental change is that the
    target-relational representation is exposed for a
    batch-hard contrastive objective during training.
    """

    def __init__(
        self,
        backbone_name: str = "roberta-base",
        num_labels: int = 3,
        relation_dim: int = 256,
        contrast_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(
            backbone_name
        )

        hidden_size = (
            self.encoder
            .config
            .hidden_size
        )

        self.relation_layer = (
            TargetConditionedRelationLayer(
                hidden_size=hidden_size,
                relation_dim=relation_dim,
                dropout=dropout,
            )
        )

        # Same target-conditioned relational pooling idea as M1.
        self.pool_query = nn.Linear(
            hidden_size,
            hidden_size,
            bias=False,
        )

        self.pool_target = nn.Linear(
            hidden_size,
            hidden_size,
            bias=False,
        )

        self.pool_norm = nn.LayerNorm(
            hidden_size
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(
                hidden_size * 2
            ),
            nn.Dropout(
                dropout
            ),
            nn.Linear(
                hidden_size * 2,
                hidden_size,
            ),
            nn.GELU(),
            nn.Dropout(
                dropout
            ),
            nn.Linear(
                hidden_size,
                num_labels,
            ),
        )

        # Small projection head used ONLY by contrastive loss.
        #
        # Classification still operates on the original M1
        # representation, so this does not replace the classifier.
        self.contrast_projector = nn.Sequential(
            nn.Linear(
                hidden_size,
                contrast_dim,
            ),
            nn.GELU(),
            nn.Linear(
                contrast_dim,
                contrast_dim,
            ),
        )

    @staticmethod
    def masked_mean(
        states,
        mask,
    ):
        weights = (
            mask
            .unsqueeze(-1)
            .to(states.dtype)
        )

        return (
            (states * weights)
            .sum(dim=1)
            / weights
            .sum(dim=1)
            .clamp_min(1.0)
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        sentence_mask=None,
        target_mask=None,
        labels=None,
        return_representation=False,
        **kwargs,
    ):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        hidden = (
            outputs
            .last_hidden_state
        )

        cls_state = hidden[:, 0]

        target_state = (
            self.masked_mean(
                hidden,
                target_mask,
            )
        )

        (
            refined,
            relation_attention,
            relation_gate,
        ) = self.relation_layer(
            sentence_states=hidden,
            target_state=target_state,
            sentence_mask=sentence_mask,
        )

        q = self.pool_query(
            refined
        )

        t = self.pool_target(
            target_state
        ).unsqueeze(1)

        pool_scores = (
            (q * t)
            .sum(dim=-1)
            / math.sqrt(
                hidden.size(-1)
            )
        )

        pool_scores = (
            pool_scores
            .masked_fill(
                ~sentence_mask.bool(),
                torch.finfo(
                    pool_scores.dtype
                ).min,
            )
        )

        pool_attention = (
            F.softmax(
                pool_scores,
                dim=-1,
            )
        )

        relational_state = (
            torch.bmm(
                pool_attention
                .unsqueeze(1),
                refined,
            )
            .squeeze(1)
        )

        relational_state = (
            self.pool_norm(
                relational_state
            )
        )

        final_state = torch.cat(
            [
                cls_state,
                relational_state,
            ],
            dim=-1,
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

        contrast_embedding = None

        if return_representation:
            contrast_embedding = (
                self.contrast_projector(
                    relational_state
                )
            )

            contrast_embedding = (
                F.normalize(
                    contrast_embedding,
                    p=2,
                    dim=-1,
                )
            )

        return RelationalSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
            contrast_embedding=contrast_embedding,
        )
