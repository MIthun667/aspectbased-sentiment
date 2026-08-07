from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.modeling_outputs import SequenceClassifierOutput

from src.models.target_evidence_trrl2 import (
    TargetEvidenceTRRL2,
)


class TRRLABSAV1(nn.Module):
    """
    TRRL-ABSA V1

    Validated TRRL-2 relation representation + global CLS.

    sentence + target
            ↓
        Pair RoBERTa
            ├──────────────→ CLS
            │
            └→ target-conditioned evidence
                    ↓
              relation embedding
                    ↓
            [CLS ; relation]
                    ↓
              sentiment classifier

    First downstream experiment intentionally uses CE only.

    This tests whether the learned relation representation itself
    provides useful information for ABSA classification.
    """

    def __init__(
        self,
        backbone_name: str = "roberta-base",
        num_labels: int = 3,
        relation_dim: int = 256,
        attention_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.relation_encoder = TargetEvidenceTRRL2(
            backbone_name=backbone_name,
            relation_dim=relation_dim,
            attention_dim=attention_dim,
            dropout=dropout,
            freeze_backbone=False,
        )

        hidden_size = (
            self.relation_encoder
            .encoder
            .config
            .hidden_size
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(
                hidden_size
                + relation_dim
            ),
            nn.Dropout(dropout),
            nn.Linear(
                hidden_size
                + relation_dim,
                hidden_size,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                hidden_size,
                num_labels,
            ),
        )

    def load_trrl2_checkpoint(
        self,
        checkpoint_path: str,
    ):
        """
        Load TRRL-2 representation-learning weights.

        Classifier remains newly initialized.
        """
        state = torch.load(
            checkpoint_path,
            map_location="cpu",
        )

        missing, unexpected = (
            self.relation_encoder
            .load_state_dict(
                state,
                strict=False,
            )
        )

        print(
            "TRRL-2 checkpoint loaded:"
        )
        print(
            checkpoint_path
        )
        print(
            "missing:",
            missing
        )
        print(
            "unexpected:",
            unexpected
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        sentence_mask=None,
        target_mask=None,
        labels=None,
        return_relation=False,
        return_evidence_attention=False,
        **kwargs,
    ):
        encoder = (
            self.relation_encoder
            .encoder
        )

        outputs = encoder(
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
            self.relation_encoder
            .masked_mean(
                hidden,
                target_mask,
            )
        )

        # ------------------------------------------
        # Target-conditioned evidence extraction
        # ------------------------------------------
        q = (
            self.relation_encoder
            .target_query(
                target_state
            )
            .unsqueeze(1)
        )

        k = (
            self.relation_encoder
            .token_key(
                hidden
            )
        )

        scores = (
            q * k
        ).sum(dim=-1)

        scores = (
            scores
            / (
                self.relation_encoder
                .attention_dim
                ** 0.5
            )
        )

        scores = scores.masked_fill(
            ~sentence_mask.bool(),
            torch.finfo(
                scores.dtype
            ).min,
        )

        evidence_attention = (
            F.softmax(
                scores,
                dim=-1,
            )
        )

        values = (
            self.relation_encoder
            .token_value(
                hidden
            )
        )

        evidence_state = (
            torch.bmm(
                evidence_attention
                .unsqueeze(1),
                values,
            )
            .squeeze(1)
        )

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
            self.relation_encoder
            .relation_projector(
                relation_input
            )
        )

        relation_embedding = (
            F.normalize(
                relation_embedding,
                p=2,
                dim=-1,
            )
        )

        # ------------------------------------------
        # Downstream fusion
        # ------------------------------------------
        fused = torch.cat(
            [
                cls_state,
                relation_embedding,
            ],
            dim=-1,
        )

        logits = self.classifier(
            fused
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

        # Training does not need these fields.
        # They are attached only when explicitly requested.
        if return_relation:
            result["relation_embedding"] = (
                relation_embedding
            )

        if return_evidence_attention:
            result["evidence_attention"] = (
                evidence_attention
            )

        return result
