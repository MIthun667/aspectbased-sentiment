from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
)


@dataclass(frozen=True, slots=True)
class TransformerClassifierOutput:
    logits: torch.Tensor
    hidden_states: tuple[torch.Tensor, ...] | None = None


class TransformerAspectClassifier(nn.Module):
    def __init__(
        self,
        *,
        backbone: nn.Module,
    ) -> None:
        super().__init__()

        self.backbone = backbone

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        *,
        number_of_classes: int = 3,
        dropout: float | None = None,
        local_files_only: bool = False,
        dtype: torch.dtype = torch.float32,
    ) -> TransformerAspectClassifier:
        if not model_name_or_path.strip():
            raise ValueError(
                "model_name_or_path must not be empty"
            )

        if number_of_classes <= 1:
            raise ValueError(
                "number_of_classes must exceed one"
            )

        if dropout is not None and not (
            0.0 <= dropout < 1.0
        ):
            raise ValueError(
                "dropout must be in [0, 1)"
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

        configuration = AutoConfig.from_pretrained(
            model_name_or_path,
            num_labels=number_of_classes,
            output_hidden_states=False,
            local_files_only=local_files_only,
        )

        if dropout is not None:
            for attribute_name in (
                "classifier_dropout",
                "hidden_dropout_prob",
                "pooler_dropout",
            ):
                if hasattr(
                    configuration,
                    attribute_name,
                ):
                    setattr(
                        configuration,
                        attribute_name,
                        dropout,
                    )

        backbone = (
            AutoModelForSequenceClassification
            .from_pretrained(
                model_name_or_path,
                config=configuration,
                local_files_only=local_files_only,
                dtype=dtype,
            )
        )

        return cls(
            backbone=backbone
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
        token_type_ids: torch.Tensor | None = None,
    ) -> TransformerClassifierOutput:
        if input_ids.ndim != 2:
            raise ValueError(
                "input_ids must have shape "
                "[batch, sequence]"
            )

        if attention_mask.shape != input_ids.shape:
            raise ValueError(
                "attention_mask must match input_ids"
            )

        if token_type_ids is not None:
            if token_type_ids.shape != input_ids.shape:
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

        output = self.backbone(
            **model_inputs
        )

        logits = getattr(
            output,
            "logits",
            None,
        )

        if not isinstance(
            logits,
            torch.Tensor,
        ):
            raise TypeError(
                "Transformer backbone did not return "
                "tensor logits"
            )

        hidden_states = getattr(
            output,
            "hidden_states",
            None,
        )

        if hidden_states is not None:
            hidden_states = tuple(
                hidden_states
            )

        return TransformerClassifierOutput(
            logits=logits,
            hidden_states=hidden_states,
        )
