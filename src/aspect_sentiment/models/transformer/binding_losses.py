from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .binding_model import (
    EvidenceBindingTransformerOutput,
)


@dataclass(frozen=True, slots=True)
class EvidenceBindingLossOutput:
    loss: torch.Tensor
    combined_loss: torch.Tensor
    context_loss: torch.Tensor
    evidence_loss: torch.Tensor
    agreement_loss: torch.Tensor
    number_with_evidence: int


class EvidenceBindingLoss(nn.Module):
    def __init__(
        self,
        *,
        combined_weight: float = 1.0,
        context_weight: float = 0.0,
        evidence_weight: float = 0.5,
        agreement_weight: float = 0.1,
        agreement_temperature: float = 1.0,
    ) -> None:
        super().__init__()

        for name, value in {
            "combined_weight": combined_weight,
            "context_weight": context_weight,
            "evidence_weight": evidence_weight,
            "agreement_weight": agreement_weight,
        }.items():
            if value < 0.0:
                raise ValueError(
                    f"{name} must be non-negative"
                )

        if agreement_temperature <= 0.0:
            raise ValueError(
                "agreement_temperature must be "
                "positive"
            )

        if (
            combined_weight
            + context_weight
            + evidence_weight
            + agreement_weight
            <= 0.0
        ):
            raise ValueError(
                "At least one loss weight must "
                "be positive"
            )

        self.combined_weight = (
            combined_weight
        )
        self.context_weight = (
            context_weight
        )
        self.evidence_weight = (
            evidence_weight
        )
        self.agreement_weight = (
            agreement_weight
        )
        self.agreement_temperature = (
            agreement_temperature
        )

    @staticmethod
    def _masked_mean(
        values: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if values.ndim != 1:
            raise ValueError(
                "values must be one-dimensional"
            )

        if mask.ndim != 1:
            raise ValueError(
                "mask must be one-dimensional"
            )

        if values.shape != mask.shape:
            raise ValueError(
                "values and mask must have "
                "matching shapes"
            )

        mask_values = mask.to(
            values.dtype
        )

        denominator = mask_values.sum()

        if float(
            denominator.detach().cpu()
        ) == 0.0:
            return values.sum() * 0.0

        return (
            values * mask_values
        ).sum() / denominator

    def forward(
        self,
        output: EvidenceBindingTransformerOutput,
        labels: torch.Tensor,
        *,
        evidence_is_empty: torch.Tensor,
    ) -> EvidenceBindingLossOutput:
        if labels.ndim != 1:
            raise ValueError(
                "labels must be one-dimensional"
            )

        batch_size = labels.shape[0]

        expected_shape = (
            batch_size,
            output.logits.shape[-1],
        )

        for name, logits in {
            "logits": output.logits,
            "context_logits": (
                output.context_logits
            ),
            "evidence_logits": (
                output.evidence_logits
            ),
        }.items():
            if logits.shape != expected_shape:
                raise ValueError(
                    f"{name} has invalid shape: "
                    f"{logits.shape}"
                )

        if evidence_is_empty.shape != (
            labels.shape
        ):
            raise ValueError(
                "evidence_is_empty must match "
                "labels"
            )

        evidence_available = (
            ~evidence_is_empty.bool()
        )

        combined_loss = (
            F.cross_entropy(
                output.logits,
                labels,
            )
        )

        context_loss = (
            F.cross_entropy(
                output.context_logits,
                labels,
            )
        )

        evidence_per_instance = (
            F.cross_entropy(
                output.evidence_logits,
                labels,
                reduction="none",
            )
        )

        evidence_loss = self._masked_mean(
            evidence_per_instance,
            evidence_available,
        )

        temperature = (
            self.agreement_temperature
        )

        combined_log_probabilities = (
            F.log_softmax(
                output.logits / temperature,
                dim=-1,
            )
        )

        evidence_probabilities = (
            F.softmax(
                output.evidence_logits
                / temperature,
                dim=-1,
            )
        )

        agreement_per_instance = (
            F.kl_div(
                combined_log_probabilities,
                evidence_probabilities,
                reduction="none",
            ).sum(dim=-1)
            * (temperature ** 2)
        )

        agreement_loss = self._masked_mean(
            agreement_per_instance,
            evidence_available,
        )

        total_loss = (
            self.combined_weight
            * combined_loss
            + self.context_weight
            * context_loss
            + self.evidence_weight
            * evidence_loss
            + self.agreement_weight
            * agreement_loss
        )

        return EvidenceBindingLossOutput(
            loss=total_loss,
            combined_loss=combined_loss,
            context_loss=context_loss,
            evidence_loss=evidence_loss,
            agreement_loss=agreement_loss,
            number_with_evidence=int(
                evidence_available.sum().item()
            ),
        )


@dataclass(frozen=True, slots=True)
class CounterfactualBindingLossOutput:
    loss: torch.Tensor
    ranking_loss: torch.Tensor
    probability_margin_loss: torch.Tensor
    number_valid: int


def masked_margin_ranking_loss(
    *,
    positive_scores: torch.Tensor,
    negative_scores: torch.Tensor,
    valid_mask: torch.Tensor,
    margin: float = 0.2,
) -> torch.Tensor:
    """
    Require the selected-evidence compatibility
    score to exceed a corrupted-evidence score.

        positive >= negative + margin
    """
    if margin < 0.0:
        raise ValueError(
            "margin must be non-negative"
        )

    if positive_scores.shape != (
        negative_scores.shape
    ):
        raise ValueError(
            "positive_scores and negative_scores "
            "must have matching shapes"
        )

    if positive_scores.ndim == 2:
        if positive_scores.shape[1] != 1:
            raise ValueError(
                "Two-dimensional scores must "
                "have shape [batch, 1]"
            )

        positive_scores = (
            positive_scores.squeeze(-1)
        )

        negative_scores = (
            negative_scores.squeeze(-1)
        )

    elif positive_scores.ndim != 1:
        raise ValueError(
            "Scores must have shape [batch] "
            "or [batch, 1]"
        )

    if valid_mask.ndim != 1:
        raise ValueError(
            "valid_mask must be one-dimensional"
        )

    if valid_mask.shape != (
        positive_scores.shape
    ):
        raise ValueError(
            "valid_mask must match score batch"
        )

    per_instance = torch.relu(
        margin
        - positive_scores
        + negative_scores
    )

    return EvidenceBindingLoss._masked_mean(
        per_instance,
        valid_mask.bool(),
    )


def masked_gold_probability_margin_loss(
    *,
    selected_logits: torch.Tensor,
    corrupted_logits: torch.Tensor,
    labels: torch.Tensor,
    valid_mask: torch.Tensor,
    margin: float = 0.05,
) -> torch.Tensor:
    """
    Require the selected-evidence prediction to
    assign more probability to the gold label than
    the corrupted-evidence prediction.

        p_gold(selected)
        >= p_gold(corrupted) + margin
    """
    if margin < 0.0:
        raise ValueError(
            "margin must be non-negative"
        )

    if selected_logits.shape != (
        corrupted_logits.shape
    ):
        raise ValueError(
            "selected_logits and corrupted_logits "
            "must have matching shapes"
        )

    if selected_logits.ndim != 2:
        raise ValueError(
            "logits must have shape "
            "[batch, classes]"
        )

    batch_size = selected_logits.shape[0]

    if labels.shape != (batch_size,):
        raise ValueError(
            "labels must have shape [batch]"
        )

    if valid_mask.shape != (batch_size,):
        raise ValueError(
            "valid_mask must have shape [batch]"
        )

    selected_probabilities = F.softmax(
        selected_logits,
        dim=-1,
    )

    corrupted_probabilities = F.softmax(
        corrupted_logits,
        dim=-1,
    )

    gold_indices = labels.unsqueeze(-1)

    selected_gold = (
        selected_probabilities.gather(
            dim=1,
            index=gold_indices,
        )
        .squeeze(-1)
    )

    corrupted_gold = (
        corrupted_probabilities.gather(
            dim=1,
            index=gold_indices,
        )
        .squeeze(-1)
    )

    per_instance = torch.relu(
        margin
        - selected_gold
        + corrupted_gold
    )

    return EvidenceBindingLoss._masked_mean(
        per_instance,
        valid_mask.bool(),
    )


class CounterfactualBindingLoss(nn.Module):
    def __init__(
        self,
        *,
        ranking_weight: float = 0.2,
        probability_margin_weight: float = 0.2,
        ranking_margin: float = 0.2,
        probability_margin: float = 0.05,
    ) -> None:
        super().__init__()

        for name, value in {
            "ranking_weight": ranking_weight,
            "probability_margin_weight": (
                probability_margin_weight
            ),
            "ranking_margin": ranking_margin,
            "probability_margin": (
                probability_margin
            ),
        }.items():
            if value < 0.0:
                raise ValueError(
                    f"{name} must be non-negative"
                )

        if (
            ranking_weight
            + probability_margin_weight
            <= 0.0
        ):
            raise ValueError(
                "At least one counterfactual "
                "loss weight must be positive"
            )

        self.ranking_weight = (
            ranking_weight
        )

        self.probability_margin_weight = (
            probability_margin_weight
        )

        self.ranking_margin = (
            ranking_margin
        )

        self.probability_margin = (
            probability_margin
        )

    def forward(
        self,
        *,
        selected_output,
        corrupted_output,
        labels: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> CounterfactualBindingLossOutput:
        ranking_loss = (
            masked_margin_ranking_loss(
                positive_scores=(
                    selected_output
                    .compatibility_score
                ),
                negative_scores=(
                    corrupted_output
                    .compatibility_score
                ),
                valid_mask=valid_mask,
                margin=self.ranking_margin,
            )
        )

        probability_margin_loss = (
            masked_gold_probability_margin_loss(
                selected_logits=(
                    selected_output.logits
                ),
                corrupted_logits=(
                    corrupted_output.logits
                ),
                labels=labels,
                valid_mask=valid_mask,
                margin=(
                    self.probability_margin
                ),
            )
        )

        total_loss = (
            self.ranking_weight
            * ranking_loss
            + self.probability_margin_weight
            * probability_margin_loss
        )

        return CounterfactualBindingLossOutput(
            loss=total_loss,
            ranking_loss=ranking_loss,
            probability_margin_loss=(
                probability_margin_loss
            ),
            number_valid=int(
                valid_mask.bool().sum().item()
            ),
        )
