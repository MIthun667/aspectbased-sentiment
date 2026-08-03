from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn.utils.rnn import (
    pack_padded_sequence,
    pad_packed_sequence,
)


@dataclass(frozen=True, slots=True)
class BiLSTMOutput:
    logits: torch.Tensor
    pooled_representation: torch.Tensor


class TargetAwareBiLSTM(nn.Module):
    def __init__(
        self,
        *,
        vocabulary_size: int,
        embedding_dimension: int,
        hidden_dimension: int,
        number_of_layers: int,
        dropout: float,
        number_of_classes: int = 3,
        padding_index: int = 0,
    ) -> None:
        super().__init__()

        if vocabulary_size <= 0:
            raise ValueError(
                "vocabulary_size must be positive"
            )

        if embedding_dimension <= 0:
            raise ValueError(
                "embedding_dimension must be positive"
            )

        if hidden_dimension <= 0:
            raise ValueError(
                "hidden_dimension must be positive"
            )

        if number_of_layers <= 0:
            raise ValueError(
                "number_of_layers must be positive"
            )

        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                "dropout must be in [0, 1)"
            )

        if number_of_classes <= 1:
            raise ValueError(
                "number_of_classes must exceed one"
            )

        self.embedding = nn.Embedding(
            num_embeddings=vocabulary_size,
            embedding_dim=embedding_dimension,
            padding_idx=padding_index,
        )

        recurrent_dropout = (
            dropout
            if number_of_layers > 1
            else 0.0
        )

        self.encoder = nn.LSTM(
            input_size=embedding_dimension,
            hidden_size=hidden_dimension,
            num_layers=number_of_layers,
            batch_first=True,
            bidirectional=True,
            dropout=recurrent_dropout,
        )

        self.dropout = nn.Dropout(dropout)

        pooled_dimension = (
            hidden_dimension * 4
        )

        self.classifier = nn.Linear(
            pooled_dimension,
            number_of_classes,
        )

    @staticmethod
    def masked_mean_pool(
        sequence_output: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).to(
            sequence_output.dtype
        )

        summed = (
            sequence_output * mask
        ).sum(dim=1)

        counts = mask.sum(dim=1).clamp_min(
            1.0
        )

        return summed / counts

    @staticmethod
    def masked_max_pool(
        sequence_output: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        expanded_mask = attention_mask.unsqueeze(
            -1
        )

        masked_output = sequence_output.masked_fill(
            ~expanded_mask,
            torch.finfo(
                sequence_output.dtype
            ).min,
        )

        pooled = masked_output.max(
            dim=1
        ).values

        empty_rows = (
            attention_mask.sum(dim=1) == 0
        )

        if empty_rows.any():
            pooled = pooled.clone()
            pooled[empty_rows] = 0.0

        return pooled

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> BiLSTMOutput:
        if input_ids.ndim != 2:
            raise ValueError(
                "input_ids must have shape "
                "[batch, sequence]"
            )

        if attention_mask.shape != input_ids.shape:
            raise ValueError(
                "attention_mask must match input_ids"
            )

        if attention_mask.dtype != torch.bool:
            raise TypeError(
                "attention_mask must use bool dtype"
            )

        lengths = attention_mask.sum(
            dim=1
        )

        if torch.any(lengths <= 0):
            raise ValueError(
                "Every input sequence must contain "
                "at least one non-padding token"
            )

        embeddings = self.embedding(
            input_ids
        )

        packed = pack_padded_sequence(
            embeddings,
            lengths=lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        packed_output, _ = self.encoder(
            packed
        )

        sequence_output, _ = (
            pad_packed_sequence(
                packed_output,
                batch_first=True,
                total_length=input_ids.shape[1],
            )
        )

        mean_pooled = self.masked_mean_pool(
            sequence_output,
            attention_mask,
        )

        max_pooled = self.masked_max_pool(
            sequence_output,
            attention_mask,
        )

        pooled = torch.cat(
            [
                mean_pooled,
                max_pooled,
            ],
            dim=-1,
        )

        logits = self.classifier(
            self.dropout(pooled)
        )

        return BiLSTMOutput(
            logits=logits,
            pooled_representation=pooled,
        )
