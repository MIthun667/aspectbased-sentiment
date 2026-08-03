from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from transformers import (
    AutoConfig,
    AutoModel,
)

from .evidence_model import masked_mean_pool


@dataclass(frozen=True, slots=True)
class EvidenceBindingTransformerOutput:
    logits: torch.Tensor
    context_logits: torch.Tensor
    evidence_logits: torch.Tensor

    context_representation: torch.Tensor
    evidence_representation: torch.Tensor
    projected_evidence_representation: (
        torch.Tensor
    )
    fused_representation: torch.Tensor

    gate_values: torch.Tensor
    evidence_available: torch.Tensor

    cls_representation: torch.Tensor
    aspect_representation: torch.Tensor
    pooled_evidence_representation: torch.Tensor
    last_hidden_state: torch.Tensor


class EvidenceBindingTransformerClassifier(
    nn.Module
):
    def __init__(
        self,
        *,
        backbone: nn.Module,
        hidden_dimension: int,
        number_of_classes: int = 3,
        dropout: float = 0.1,
        gate_dimension: int | None = None,
    ) -> None:
        super().__init__()

        if hidden_dimension <= 0:
            raise ValueError(
                "hidden_dimension must be positive"
            )

        if number_of_classes <= 1:
            raise ValueError(
                "number_of_classes must exceed one"
            )

        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                "dropout must be in [0, 1)"
            )

        resolved_gate_dimension = (
            hidden_dimension
            if gate_dimension is None
            else gate_dimension
        )

        if resolved_gate_dimension <= 0:
            raise ValueError(
                "gate_dimension must be positive"
            )

        self.backbone = backbone
        self.hidden_dimension = (
            hidden_dimension
        )
        self.number_of_classes = (
            number_of_classes
        )
        self.gate_dimension = (
            resolved_gate_dimension
        )

        context_input_dimension = (
            hidden_dimension * 2
        )

        evidence_input_dimension = (
            hidden_dimension * 2 + 1
        )

        gate_input_dimension = (
            hidden_dimension * 3 + 1
        )

        self.dropout = nn.Dropout(dropout)

        self.context_projection = nn.Sequential(
            nn.Linear(
                context_input_dimension,
                hidden_dimension,
            ),
            nn.GELU(),
            nn.LayerNorm(hidden_dimension),
        )

        self.evidence_projection = nn.Sequential(
            nn.Linear(
                evidence_input_dimension,
                hidden_dimension,
            ),
            nn.GELU(),
            nn.LayerNorm(hidden_dimension),
        )

        self.gate_network = nn.Sequential(
            nn.Linear(
                gate_input_dimension,
                resolved_gate_dimension,
            ),
            nn.GELU(),
            nn.Linear(
                resolved_gate_dimension,
                hidden_dimension,
            ),
            nn.Sigmoid(),
        )

        self.context_classifier = nn.Linear(
            hidden_dimension,
            number_of_classes,
        )

        self.evidence_classifier = nn.Linear(
            hidden_dimension,
            number_of_classes,
        )

        self.combined_classifier = nn.Linear(
            hidden_dimension,
            number_of_classes,
        )

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        *,
        number_of_classes: int = 3,
        dropout: float = 0.1,
        gate_dimension: int | None = None,
        local_files_only: bool = False,
        dtype: torch.dtype = torch.float32,
    ) -> EvidenceBindingTransformerClassifier:
        if not model_name_or_path.strip():
            raise ValueError(
                "model_name_or_path must not be empty"
            )

        if dtype not in {
            torch.float32,
            torch.bfloat16,
            torch.float16,
        }:
            raise ValueError(
                "dtype must be torch.float32, "
                "torch.bfloat16, or torch.float16"
            )

        configuration = (
            AutoConfig.from_pretrained(
                model_name_or_path,
                output_hidden_states=False,
                local_files_only=(
                    local_files_only
                ),
            )
        )

        hidden_dimension = getattr(
            configuration,
            "hidden_size",
            None,
        )

        if not isinstance(
            hidden_dimension,
            int,
        ):
            raise TypeError(
                "Transformer configuration "
                "does not define hidden_size"
            )

        backbone = AutoModel.from_pretrained(
            model_name_or_path,
            config=configuration,
            local_files_only=(
                local_files_only
            ),
            dtype=dtype,
        )

        return cls(
            backbone=backbone,
            hidden_dimension=hidden_dimension,
            number_of_classes=(
                number_of_classes
            ),
            dropout=dropout,
            gate_dimension=gate_dimension,
        )

    @property
    def configuration(self) -> Any:
        configuration = getattr(
            self.backbone,
            "config",
            None,
        )

        if configuration is None:
            raise AttributeError(
                "Transformer backbone has no config"
            )

        return configuration

    @staticmethod
    def _validate_mask(
        mask: torch.Tensor | None,
        *,
        input_ids: torch.Tensor,
        name: str,
    ) -> torch.Tensor:
        if mask is None:
            raise ValueError(
                f"{name} is required"
            )

        if mask.shape != input_ids.shape:
            raise ValueError(
                f"{name} must match input_ids"
            )

        return mask

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        aspect_subword_mask: torch.Tensor,
        evidence_subword_mask: torch.Tensor,
        token_type_ids: (
            torch.Tensor | None
        ) = None,
    ) -> EvidenceBindingTransformerOutput:
        if input_ids.ndim != 2:
            raise ValueError(
                "input_ids must have shape "
                "[batch, sequence]"
            )

        if attention_mask.shape != (
            input_ids.shape
        ):
            raise ValueError(
                "attention_mask must match input_ids"
            )

        aspect_subword_mask = (
            self._validate_mask(
                aspect_subword_mask,
                input_ids=input_ids,
                name="aspect_subword_mask",
            )
        )

        evidence_subword_mask = (
            self._validate_mask(
                evidence_subword_mask,
                input_ids=input_ids,
                name="evidence_subword_mask",
            )
        )

        if token_type_ids is not None:
            if token_type_ids.shape != (
                input_ids.shape
            ):
                raise ValueError(
                    "token_type_ids must match input_ids"
                )

        model_inputs: dict[
            str,
            torch.Tensor,
        ] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        if token_type_ids is not None:
            model_inputs["token_type_ids"] = (
                token_type_ids
            )

        backbone_output = self.backbone(
            **model_inputs
        )

        last_hidden_state = getattr(
            backbone_output,
            "last_hidden_state",
            None,
        )

        if not isinstance(
            last_hidden_state,
            torch.Tensor,
        ):
            raise TypeError(
                "Transformer backbone did not "
                "return last_hidden_state"
            )

        if last_hidden_state.ndim != 3:
            raise ValueError(
                "last_hidden_state must have shape "
                "[batch, sequence, hidden]"
            )

        if last_hidden_state.shape[:2] != (
            input_ids.shape
        ):
            raise ValueError(
                "last_hidden_state sequence shape "
                "does not match input_ids"
            )

        cls_representation = (
            last_hidden_state[:, 0]
        )

        aspect_representation = (
            masked_mean_pool(
                last_hidden_state,
                aspect_subword_mask,
            )
        )

        pooled_evidence_representation = (
            masked_mean_pool(
                last_hidden_state,
                evidence_subword_mask,
            )
        )

        evidence_available = (
            evidence_subword_mask.any(
                dim=1,
                keepdim=True,
            )
            .to(last_hidden_state.dtype)
        )

        context_input = torch.cat(
            [
                cls_representation,
                aspect_representation,
            ],
            dim=-1,
        )

        evidence_input = torch.cat(
            [
                aspect_representation,
                pooled_evidence_representation,
                evidence_available,
            ],
            dim=-1,
        )

        context_representation = (
            self.context_projection(
                self.dropout(
                    context_input
                )
            )
        )

        evidence_representation = (
            self.evidence_projection(
                self.dropout(
                    evidence_input
                )
            )
        )

        evidence_representation = (
            evidence_representation
            * evidence_available
        )

        gate_input = torch.cat(
            [
                cls_representation,
                aspect_representation,
                pooled_evidence_representation,
                evidence_available,
            ],
            dim=-1,
        )

        gate_values = self.gate_network(
            self.dropout(gate_input)
        )

        gate_values = (
            gate_values
            * evidence_available
        )

        projected_evidence_representation = (
            gate_values
            * evidence_representation
        )

        fused_representation = (
            context_representation
            + projected_evidence_representation
        )

        context_logits = (
            self.context_classifier(
                self.dropout(
                    context_representation
                )
            )
        )

        evidence_logits = (
            self.evidence_classifier(
                self.dropout(
                    evidence_representation
                )
            )
        )

        logits = self.combined_classifier(
            self.dropout(
                fused_representation
            )
        )

        return EvidenceBindingTransformerOutput(
            logits=logits,
            context_logits=context_logits,
            evidence_logits=evidence_logits,
            context_representation=(
                context_representation
            ),
            evidence_representation=(
                evidence_representation
            ),
            projected_evidence_representation=(
                projected_evidence_representation
            ),
            fused_representation=(
                fused_representation
            ),
            gate_values=gate_values,
            evidence_available=(
                evidence_available
            ),
            cls_representation=(
                cls_representation
            ),
            aspect_representation=(
                aspect_representation
            ),
            pooled_evidence_representation=(
                pooled_evidence_representation
            ),
            last_hidden_state=(
                last_hidden_state
            ),
        )
