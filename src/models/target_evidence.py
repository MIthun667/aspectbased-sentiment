from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel
from transformers.modeling_outputs import SequenceClassifierOutput


@dataclass
class TargetEvidenceOutput(SequenceClassifierOutput):
    evidence_attentions: torch.FloatTensor | None = None


class TargetEvidenceModel(nn.Module):
    """
    Target-conditioned evidence-binding model.

    Pipeline:
        Transformer token representations
                ↓
        target span pooling
                ↓
        target-token compatibility
                +
        dependency-distance embedding
                ↓
        target-conditioned evidence attention
                ↓
        evidence representation
                ↓
        CLS + target + evidence + interaction
                ↓
        sentiment classifier
    """

    def __init__(
        self,
        backbone_name: str = "roberta-base",
        num_labels: int = 3,
        max_dependency_distance: int = 10,
        distance_embedding_dim: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.backbone_name = backbone_name
        self.num_labels = num_labels
        self.max_dependency_distance = max_dependency_distance

        self.encoder = AutoModel.from_pretrained(backbone_name)

        hidden_size = self.encoder.config.hidden_size

        self.distance_embedding = nn.Embedding(
            max_dependency_distance + 2,
            distance_embedding_dim,
        )

        # Score each contextual token conditioned on the target.
        #
        # Features:
        #   token hidden state
        #   target hidden state
        #   elementwise interaction
        #   absolute difference
        #   dependency-distance embedding
        evidence_input_dim = (
            hidden_size * 4
            + distance_embedding_dim
        )

        self.evidence_scorer = nn.Sequential(
            nn.LayerNorm(evidence_input_dim),
            nn.Linear(evidence_input_dim, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

        # Final representation:
        #   CLS
        #   target
        #   evidence
        #   target * evidence
        fusion_dim = hidden_size * 4

        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_labels),
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        aspect_mask=None,
        distance_ids=None,
        evidence_mask=None,
        labels=None,
        **kwargs,
    ):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        hidden = outputs.last_hidden_state
        # [B, L, H]

        cls_repr = hidden[:, 0]
        # [B, H]

        # ---------------------------------------------------------
        # Target representation
        # ---------------------------------------------------------

        aspect_weights = aspect_mask.float()

        target_denominator = (
            aspect_weights.sum(dim=1, keepdim=True)
            .clamp_min(1.0)
        )

        target_repr = (
            hidden * aspect_weights.unsqueeze(-1)
        ).sum(dim=1) / target_denominator

        # [B, H]

        # ---------------------------------------------------------
        # Target-conditioned token features
        # ---------------------------------------------------------

        target_expanded = target_repr.unsqueeze(1).expand_as(hidden)

        interaction = hidden * target_expanded
        difference = torch.abs(hidden - target_expanded)

        clipped_distances = torch.clamp(
            distance_ids,
            min=0,
            max=self.max_dependency_distance + 1,
        )

        distance_repr = self.distance_embedding(
            clipped_distances
        )

        evidence_features = torch.cat(
            [
                hidden,
                target_expanded,
                interaction,
                difference,
                distance_repr,
            ],
            dim=-1,
        )

        evidence_logits = (
            self.evidence_scorer(evidence_features)
            .squeeze(-1)
        )

        # Mask special/padding tokens.
        mask = evidence_mask.bool()

        evidence_logits = evidence_logits.masked_fill(
            ~mask,
            -1e4,
        )

        evidence_attention = F.softmax(
            evidence_logits,
            dim=-1,
        )

        evidence_repr = torch.sum(
            hidden * evidence_attention.unsqueeze(-1),
            dim=1,
        )

        # ---------------------------------------------------------
        # Target/evidence fusion
        # ---------------------------------------------------------

        fusion = torch.cat(
            [
                cls_repr,
                target_repr,
                evidence_repr,
                target_repr * evidence_repr,
            ],
            dim=-1,
        )

        logits = self.classifier(fusion)

        loss = None

        if labels is not None:
            loss = F.cross_entropy(
                logits,
                labels,
            )

        return TargetEvidenceOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            evidence_attentions=evidence_attention,
        )

