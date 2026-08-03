from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import nn
from transformers import (
    AutoConfig,
    AutoModel,
)


EvidencePoolingMode = Literal[
    "cls",
    "cls_aspect",
    "cls_aspect_evidence",
]

SUPPORTED_EVIDENCE_POOLING_MODES = frozenset(
    {
        "cls",
        "cls_aspect",
        "cls_aspect_evidence",
    }
)


@dataclass(frozen=True, slots=True)
class EvidenceTransformerClassifierOutput:
    logits: torch.Tensor
    cls_representation: torch.Tensor
    aspect_representation: torch.Tensor | None
    evidence_representation: torch.Tensor | None
    evidence_available: torch.Tensor | None
    last_hidden_state: torch.Tensor


def masked_mean_pool(
    hidden_states: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if hidden_states.ndim != 3:
        raise ValueError(
            "hidden_states must have shape "
            "[batch, sequence, hidden]"
        )

    if mask.ndim != 2:
        raise ValueError(
            "mask must have shape "
            "[batch, sequence]"
        )

    if mask.shape != hidden_states.shape[:2]:
        raise ValueError(
            "mask must match the first two "
            "hidden-state dimensions"
        )

    mask_values = (
        mask.unsqueeze(-1)
        .to(hidden_states.dtype)
    )

    numerator = (
        hidden_states * mask_values
    ).sum(dim=1)

    denominator = mask_values.sum(
        dim=1
    ).clamp_min(1.0)

    return numerator / denominator


class EvidenceAwareTransformerClassifier(
    nn.Module
):
    def __init__(
        self,
        *,
        backbone: nn.Module,
        hidden_dimension: int,
        number_of_classes: int = 3,
        mode: EvidencePoolingMode = (
            "cls_aspect_evidence"
        ),
        dropout: float = 0.1,
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

        if mode not in (
            SUPPORTED_EVIDENCE_POOLING_MODES
        ):
            raise ValueError(
                f"Unsupported evidence mode: {mode!r}"
            )

        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                "dropout must be in [0, 1)"
            )

        self.backbone = backbone
        self.hidden_dimension = (
            hidden_dimension
        )
        self.number_of_classes = (
            number_of_classes
        )
        self.mode = mode

        representation_multiplier = {
            "cls": 1,
            "cls_aspect": 2,
            "cls_aspect_evidence": 3,
        }[mode]

        indicator_dimension = (
            1
            if mode
            == "cls_aspect_evidence"
            else 0
        )

        classifier_input_dimension = (
            hidden_dimension
            * representation_multiplier
            + indicator_dimension
        )

        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Linear(
            classifier_input_dimension,
            number_of_classes,
        )

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        *,
        number_of_classes: int = 3,
        mode: EvidencePoolingMode = (
            "cls_aspect_evidence"
        ),
        dropout: float = 0.1,
        local_files_only: bool = False,
        dtype: torch.dtype = torch.float32,
    ) -> EvidenceAwareTransformerClassifier:
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
            mode=mode,
            dropout=dropout,
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

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        aspect_subword_mask: (
            torch.Tensor | None
        ) = None,
        evidence_subword_mask: (
            torch.Tensor | None
        ) = None,
        token_type_ids: (
            torch.Tensor | None
        ) = None,
    ) -> EvidenceTransformerClassifierOutput:
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

        if token_type_ids is not None:
            if token_type_ids.shape != (
                input_ids.shape
            ):
                raise ValueError(
                    "token_type_ids must match input_ids"
                )

        if self.mode in {
            "cls_aspect",
            "cls_aspect_evidence",
        }:
            if aspect_subword_mask is None:
                raise ValueError(
                    "aspect_subword_mask is required "
                    f"for mode {self.mode!r}"
                )

            if aspect_subword_mask.shape != (
                input_ids.shape
            ):
                raise ValueError(
                    "aspect_subword_mask must "
                    "match input_ids"
                )

        if self.mode == (
            "cls_aspect_evidence"
        ):
            if evidence_subword_mask is None:
                raise ValueError(
                    "evidence_subword_mask is "
                    "required for evidence mode"
                )

            if evidence_subword_mask.shape != (
                input_ids.shape
            ):
                raise ValueError(
                    "evidence_subword_mask must "
                    "match input_ids"
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
                "last_hidden_state must have "
                "shape [batch, sequence, hidden]"
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

        pooled_representations = [
            cls_representation
        ]

        aspect_representation = None
        evidence_representation = None
        evidence_available = None

        if self.mode in {
            "cls_aspect",
            "cls_aspect_evidence",
        }:
            assert (
                aspect_subword_mask
                is not None
            )

            aspect_representation = (
                masked_mean_pool(
                    last_hidden_state,
                    aspect_subword_mask,
                )
            )

            pooled_representations.append(
                aspect_representation
            )

        if self.mode == (
            "cls_aspect_evidence"
        ):
            assert (
                evidence_subword_mask
                is not None
            )

            evidence_representation = (
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
                .to(
                    last_hidden_state.dtype
                )
            )

            pooled_representations.extend(
                [
                    evidence_representation,
                    evidence_available,
                ]
            )

        combined = torch.cat(
            pooled_representations,
            dim=-1,
        )

        logits = self.classifier(
            self.dropout(combined)
        )

        return (
            EvidenceTransformerClassifierOutput(
                logits=logits,
                cls_representation=(
                    cls_representation
                ),
                aspect_representation=(
                    aspect_representation
                ),
                evidence_representation=(
                    evidence_representation
                ),
                evidence_available=(
                    evidence_available
                ),
                last_hidden_state=(
                    last_hidden_state
                ),
            )
        )
