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
