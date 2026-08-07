from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel
from transformers.modeling_outputs import SequenceClassifierOutput


class TargetConditionedRelationLayer(nn.Module):
    """
    Learns target-conditioned token-to-token relations.

    For sentence token states h_i and target representation h_t:

        q_i = W_q h_i
        k_j = W_k h_j
        v_j = W_v h_j
        t   = W_t h_t

        score_ij =
            ((q_i * t) dot k_j) / sqrt(d)

        A_ij = softmax_j(score_ij)

        r_i = sum_j A_ij v_j

    A gated residual update then produces refined token states.
    """

    def __init__(
        self,
        hidden_size: int,
        relation_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.relation_dim = relation_dim

        self.query = nn.Linear(hidden_size, relation_dim)
        self.key = nn.Linear(hidden_size, relation_dim)
        self.value = nn.Linear(hidden_size, relation_dim)
        self.target = nn.Linear(hidden_size, relation_dim)

        self.output = nn.Linear(relation_dim, hidden_size)

        self.gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid(),
        )

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        sentence_states: torch.Tensor,
        target_state: torch.Tensor,
        sentence_mask: torch.Tensor,
    ):
        """
        sentence_states:
            [B, L, H]

        target_state:
            [B, H]

        sentence_mask:
            [B, L], 1 for sentence tokens, 0 otherwise.
        """

        q = self.query(sentence_states)
        k = self.key(sentence_states)
        v = self.value(sentence_states)

        target = self.target(target_state).unsqueeze(1)

        # Target-conditioned queries.
        target_q = q * target

        scores = torch.matmul(
            target_q,
            k.transpose(-1, -2),
        )

        scores = scores / math.sqrt(self.relation_dim)

        # Mask candidate relation destinations.
        key_mask = sentence_mask.unsqueeze(1).bool()

        scores = scores.masked_fill(
            ~key_mask,
            torch.finfo(scores.dtype).min,
        )

        relation_attention = F.softmax(scores, dim=-1)

        # Remove padded/source-invalid rows as well.
        query_mask = sentence_mask.unsqueeze(-1).to(
            relation_attention.dtype
        )

        relation_attention = (
            relation_attention
            * sentence_mask.unsqueeze(1).to(
                relation_attention.dtype
            )
        )

        relation_attention = relation_attention / (
            relation_attention.sum(
                dim=-1,
                keepdim=True,
            ).clamp_min(1e-8)
        )

        relational = torch.matmul(
            relation_attention,
            v,
        )

        relational = self.output(relational)
        relational = self.dropout(relational)

        target_expanded = target_state.unsqueeze(1).expand_as(
            sentence_states
        )

        gate_input = torch.cat(
            [sentence_states, target_expanded],
            dim=-1,
        )

        gate = self.gate(gate_input)

        refined = self.layer_norm(
            sentence_states
            + gate * relational
        )

        refined = refined * query_mask

        return refined, relation_attention, gate


class TargetRelationalM1(nn.Module):
    """
    M1:
        Pair-RoBERTa
          +
        learned target-conditioned relational refinement.

    No dependency graph.
    No semantic threshold graph.
    No contrastive objective.
    No capsule routing.

    This experiment isolates the value of learned
    target-conditioned relational structure.
    """

    def __init__(
        self,
        backbone_name: str = "roberta-base",
        num_labels: int = 3,
        relation_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.backbone_name = backbone_name
        self.num_labels = num_labels

        self.encoder = AutoModel.from_pretrained(
            backbone_name
        )

        hidden_size = self.encoder.config.hidden_size

        self.relation_layer = TargetConditionedRelationLayer(
            hidden_size=hidden_size,
            relation_dim=relation_dim,
            dropout=dropout,
        )

        # Target-conditioned pooling over refined sentence states.
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

        self.pool_norm = nn.LayerNorm(hidden_size)

        # Keep CLS as a strong baseline path while allowing
        # relational features to contribute.
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_labels),
        )

    @staticmethod
    def masked_mean(
        states: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:

        weights = mask.unsqueeze(-1).to(states.dtype)

        summed = (states * weights).sum(dim=1)

        denominator = weights.sum(dim=1).clamp_min(1.0)

        return summed / denominator

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

        # Target representation comes from the appended second sequence.
        target_state = self.masked_mean(
            hidden,
            target_mask,
        )

        refined, relation_attention, gate = (
            self.relation_layer(
                sentence_states=hidden,
                target_state=target_state,
                sentence_mask=sentence_mask,
            )
        )

        # Target-conditioned pooling.
        q = self.pool_query(refined)

        t = self.pool_target(
            target_state
        ).unsqueeze(1)

        pool_scores = (
            q * t
        ).sum(dim=-1) / math.sqrt(
            hidden.size(-1)
        )

        pool_scores = pool_scores.masked_fill(
            ~sentence_mask.bool(),
            torch.finfo(pool_scores.dtype).min,
        )

        pool_attention = F.softmax(
            pool_scores,
            dim=-1,
        )

        relational_state = torch.bmm(
            pool_attention.unsqueeze(1),
            refined,
        ).squeeze(1)

        relational_state = self.pool_norm(
            relational_state
        )

        final_state = torch.cat(
            [
                cls_state,
                relational_state,
            ],
            dim=-1,
        )

        logits = self.classifier(final_state)

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

        # Diagnostics are attached without interfering with Trainer.
        result.relation_attention = relation_attention
        result.pool_attention = pool_attention
        result.relation_gate = gate

        return result
