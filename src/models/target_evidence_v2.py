from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel
from transformers.modeling_outputs import SequenceClassifierOutput


@dataclass
class TargetEvidenceV2Output(SequenceClassifierOutput):
    evidence_attentions: torch.FloatTensor | None = None
    gate_values: torch.FloatTensor | None = None


class TargetEvidenceV2Model(nn.Module):
    """
    Target-Evidence V2-A: Gated Residual Evidence Refinement over Sequence-Pair RoBERTa Baseline.

    Pipeline:
        Sequence-pair RoBERTa (<s> sentence </s></s> aspect </s>)
                ↓
        Last hidden states [B, L, H]
                ↓
        h_cls = hidden[:, 0] (Sequence-pair baseline representation)
        h_target = mean pool over sentence target tokens
                ↓
        Target-conditioned evidence attention over SENTENCE tokens only
        (excluding second-sequence aspect tokens & special/padding tokens)
                ↓
        h_evidence = sum_i alpha_i h_i
                ↓
        Gated Residual Fusion:
          e = W_e(h_evidence)
          gate = sigmoid(W_g([h_cls ; h_target ; h_evidence]))
          h_final = h_cls + gate * e
                ↓
        Classifier(h_final)
    """

    def __init__(
        self,
        backbone_name: str = "roberta-base",
        num_labels: int = 3,
        max_dependency_distance: int = 10,
        distance_embedding_dim: int = 32,
        use_distance_embedding: bool = True,
        use_evidence_supervision: bool = False,
        lambda_evid: float = 0.1,
        gamma_tdc: float = 0.4,
        lambda_supp: float = 1.0,
        dropout: float = 0.1,
        initial_gate_bias: float = -2.0,
    ):
        super().__init__()

        self.backbone_name = backbone_name
        self.num_labels = num_labels
        self.max_dependency_distance = max_dependency_distance
        self.use_distance_embedding = use_distance_embedding
        self.use_evidence_supervision = use_evidence_supervision
        self.lambda_evid = lambda_evid
        self.gamma_tdc = gamma_tdc
        self.lambda_supp = lambda_supp

        self.encoder = AutoModel.from_pretrained(backbone_name)
        hidden_size = self.encoder.config.hidden_size

        if use_distance_embedding:
            self.distance_embedding = nn.Embedding(
                max_dependency_distance + 2,
                distance_embedding_dim,
            )
            evidence_input_dim = hidden_size * 4 + distance_embedding_dim
        else:
            self.distance_embedding = None
            evidence_input_dim = hidden_size * 4

        # Efficient bottleneck evidence scorer
        self.evidence_scorer = nn.Sequential(
            nn.LayerNorm(evidence_input_dim),
            nn.Linear(evidence_input_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

        # Residual projection
        self.evidence_proj = nn.Linear(hidden_size, hidden_size)

        # Gated fusion generator: [h_cls ; h_target ; h_evidence] -> gate_dim (hidden_size)
        self.gate_generator = nn.Sequential(
            nn.Linear(hidden_size * 3, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, hidden_size),
        )

        # Conservative initialization of gate bias
        nn.init.constant_(self.gate_generator[-1].bias, initial_gate_bias)
        nn.init.normal_(self.gate_generator[-1].weight, std=0.01)

        # Final Classifier on h_final = h_cls + gate * e
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
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
        # Target representation (pooled over sentence target tokens)
        # ---------------------------------------------------------
        aspect_weights = aspect_mask.float()
        target_denominator = aspect_weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        target_repr = (hidden * aspect_weights.unsqueeze(-1)).sum(dim=1) / target_denominator
        # [B, H]

        # ---------------------------------------------------------
        # Evidence Scorer over Sentence Tokens
        # ---------------------------------------------------------
        target_expanded = target_repr.unsqueeze(1).expand_as(hidden)
        interaction = hidden * target_expanded
        difference = torch.abs(hidden - target_expanded)

        feature_list = [hidden, target_expanded, interaction, difference]

        if self.use_distance_embedding and distance_ids is not None:
            clipped_distances = torch.clamp(
                distance_ids,
                min=0,
                max=self.max_dependency_distance + 1,
            )
            distance_repr = self.distance_embedding(clipped_distances)
            feature_list.append(distance_repr)

        evidence_features = torch.cat(feature_list, dim=-1)

        evidence_logits = self.evidence_scorer(evidence_features).squeeze(-1)

        # Mask non-evidence tokens (aspect in sequence 1, special, padding tokens)
        mask = evidence_mask.bool()
        evidence_logits = evidence_logits.masked_fill(~mask, -1e4)

        evidence_attention = F.softmax(evidence_logits, dim=-1)
        evidence_repr = torch.sum(hidden * evidence_attention.unsqueeze(-1), dim=1)
        # [B, H]

        # ---------------------------------------------------------
        # Gated Residual Fusion
        # ---------------------------------------------------------
        refinement = self.evidence_proj(evidence_repr)

        gate_input = torch.cat([cls_repr, target_repr, evidence_repr], dim=-1)
        gate_logits = self.gate_generator(gate_input)
        gate = torch.sigmoid(gate_logits)

        h_final = cls_repr + gate * refinement

        logits = self.classifier(h_final)

        loss = None
        if labels is not None:
            loss_ce = F.cross_entropy(logits, labels)

            if self.use_evidence_supervision:
                # 1. Target-Span Suppression Loss (L_SUPP)
                asp_mass = (evidence_attention * aspect_weights).sum(dim=-1)
                loss_supp = asp_mass.mean()

                # 2. Target-Disambiguation Contrastive Loss (L_TDC)
                loss_tdc = torch.tensor(0.0, device=logits.device)
                pair_count = 0
                batch_size = input_ids.size(0)

                for i in range(batch_size):
                    for j in range(i + 1, batch_size):
                        if torch.equal(input_ids[i], input_ids[j]) and not torch.equal(aspect_mask[i], aspect_mask[j]):
                            p = evidence_attention[i]
                            q = evidence_attention[j]
                            m = 0.5 * (p + q)
                            eps = 1e-12
                            kld_p = (p * (torch.log(p + eps) - torch.log(m + eps))).sum()
                            kld_q = (q * (torch.log(q + eps) - torch.log(m + eps))).sum()
                            jsd = 0.5 * kld_p + 0.5 * kld_q

                            loss_pair = F.relu(self.gamma_tdc - jsd)
                            loss_tdc = loss_tdc + loss_pair
                            pair_count += 1

                if pair_count > 0:
                    loss_tdc = loss_tdc / pair_count

                loss_evid = self.lambda_supp * loss_supp + loss_tdc
                loss = loss_ce + self.lambda_evid * loss_evid
            else:
                loss = loss_ce

        return TargetEvidenceV2Output(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            evidence_attentions=evidence_attention,
            gate_values=gate,
        )

