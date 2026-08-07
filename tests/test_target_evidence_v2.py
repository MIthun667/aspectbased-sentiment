from __future__ import annotations

import pytest
import torch
from transformers import AutoTokenizer

from src.models.target_evidence_v2 import TargetEvidenceV2Model


def build_v2_inputs(
    tokenizer: AutoTokenizer,
    sentence_tokens: list[str],
    aspect_tokens: list[str],
    aspect_start: int,
    aspect_end: int,
    aspect_distances: list[int],
    max_length: int = 128,
    max_dependency_distance: int = 10,
):
    encoding = tokenizer(
        sentence_tokens,
        aspect_tokens,
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_attention_mask=True,
    )

    sequence_ids = encoding.sequence_ids()
    word_ids = encoding.word_ids()

    aspect_mask = []
    distance_ids = []
    evidence_mask = []

    special_distance = max_dependency_distance + 1

    for seq_id, word_id in zip(sequence_ids, word_ids):
        # Sentence tokens are sequence_id == 0 and word_id is not None
        if seq_id == 0 and word_id is not None:
            is_target = (aspect_start <= word_id < aspect_end)
            aspect_mask.append(1 if is_target else 0)
            
            dist = aspect_distances[word_id] if word_id < len(aspect_distances) else special_distance
            dist = min(int(dist), max_dependency_distance)
            distance_ids.append(dist)
            
            evidence_mask.append(1)  # Real sentence subwords are valid evidence
        else:
            # Special tokens, padding, or second-sequence aspect tokens
            aspect_mask.append(0)
            distance_ids.append(special_distance)
            evidence_mask.append(0)  # Exclude from evidence attention

    return {
        "input_ids": torch.tensor([encoding["input_ids"]], dtype=torch.long),
        "attention_mask": torch.tensor([encoding["attention_mask"]], dtype=torch.long),
        "aspect_mask": torch.tensor([aspect_mask], dtype=torch.long),
        "distance_ids": torch.tensor([distance_ids], dtype=torch.long),
        "evidence_mask": torch.tensor([evidence_mask], dtype=torch.long),
        "sequence_ids": sequence_ids,
        "word_ids": word_ids,
    }


def test_sequence_pair_tokenization_and_masks():
    tokenizer = AutoTokenizer.from_pretrained("roberta-base", add_prefix_space=True)
    sentence = ["The", "screen", "is", "excellent", "but", "the", "battery", "is", "terrible", "."]
    aspect = ["screen"]
    aspect_distances = [0, 0, 1, 2, 3, 2, 4, 5, 6, 7]

    inputs = build_v2_inputs(
        tokenizer=tokenizer,
        sentence_tokens=sentence,
        aspect_tokens=aspect,
        aspect_start=1,
        aspect_end=2,
        aspect_distances=aspect_distances,
    )

    # 1. Check sequence_ids and exclusion of second sequence aspect
    seq_ids = inputs["sequence_ids"]
    word_ids = inputs["word_ids"]
    ev_mask = inputs["evidence_mask"][0].numpy()
    asp_mask = inputs["aspect_mask"][0].numpy()

    # Second sequence aspect tokens (seq_id == 1) must be 0 in evidence_mask
    for i, (seq_id, wid) in enumerate(zip(seq_ids, word_ids)):
        if seq_id != 0 or wid is None:
            assert ev_mask[i] == 0

    # 2. Target aspect mask in sentence (seq_id == 0, word_id == 1 ("screen")) must be 1
    assert asp_mask.sum() > 0


def test_multiword_aspect_alignment():
    tokenizer = AutoTokenizer.from_pretrained("roberta-base", add_prefix_space=True)
    sentence = ["tech", "support", "would", "not", "fix", "the", "problem", "."]
    aspect = ["tech", "support"]
    aspect_distances = [0, 0, 2, 2, 1, 2, 3, 4]

    inputs = build_v2_inputs(
        tokenizer=tokenizer,
        sentence_tokens=sentence,
        aspect_tokens=aspect,
        aspect_start=0,
        aspect_end=2,
        aspect_distances=aspect_distances,
    )

    asp_mask = inputs["aspect_mask"][0]
    assert asp_mask.sum().item() >= 2  # Both "tech" and "support" subwords masked


def test_target_evidence_v2_model_forward():
    model = TargetEvidenceV2Model(
        backbone_name="roberta-base",
        num_labels=3,
        use_distance_embedding=True,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("roberta-base", add_prefix_space=True)
    sentence = ["The", "battery", "life", "is", "great", "."]
    aspect = ["battery", "life"]
    distances = [1, 0, 0, 1, 2, 3]

    inputs = build_v2_inputs(
        tokenizer=tokenizer,
        sentence_tokens=sentence,
        aspect_tokens=aspect,
        aspect_start=1,
        aspect_end=3,
        aspect_distances=distances,
    )

    labels = torch.tensor([2], dtype=torch.long)

    with torch.no_grad():
        output = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            aspect_mask=inputs["aspect_mask"],
            distance_ids=inputs["distance_ids"],
            evidence_mask=inputs["evidence_mask"],
            labels=labels,
        )

    # Assertions
    assert output.loss is not None
    assert not torch.isnan(output.loss)
    assert not torch.isinf(output.loss)
    assert output.logits.shape == (1, 3)

    # Evidence attentions sum to 1.0
    attn_sum = output.evidence_attentions.sum(dim=-1).item()
    assert pytest.approx(attn_sum, abs=1e-4) == 1.0

    # Gate values are between 0 and 1
    gates = output.gate_values
    assert torch.all(gates >= 0.0) and torch.all(gates <= 1.0)


def test_different_targets_same_sentence():
    tokenizer = AutoTokenizer.from_pretrained("roberta-base", add_prefix_space=True)
    sentence = ["The", "screen", "is", "great", "but", "battery", "is", "bad", "."]

    # Target 1: screen
    inputs1 = build_v2_inputs(
        tokenizer, sentence, ["screen"], 1, 2, [1, 0, 1, 2, 3, 4, 5, 6, 7]
    )
    # Target 2: battery
    inputs2 = build_v2_inputs(
        tokenizer, sentence, ["battery"], 5, 6, [4, 3, 2, 1, 1, 0, 1, 2, 3]
    )

    assert inputs1["aspect_mask"].sum().item() > 0
    assert inputs2["aspect_mask"].sum().item() > 0
    # The two target masks must be different
    assert not torch.equal(inputs1["aspect_mask"], inputs2["aspect_mask"])


def test_truncation_detection():
    tokenizer = AutoTokenizer.from_pretrained("roberta-base", add_prefix_space=True)
    # Create very long sentence where aspect is beyond max_length 16
    sentence = ["word"] * 30 + ["target_aspect"]
    aspect = ["target_aspect"]
    distances = [10] * 31

    inputs = build_v2_inputs(
        tokenizer, sentence, aspect, 30, 31, distances, max_length=16
    )

    # Truncation detection
    assert inputs["aspect_mask"].sum().item() == 0


def test_evidence_supervision_loss():
    model = TargetEvidenceV2Model(
        backbone_name="roberta-base",
        num_labels=3,
        use_evidence_supervision=True,
        lambda_evid=0.1,
        gamma_tdc=0.4,
    )
    model.train()

    tokenizer = AutoTokenizer.from_pretrained("roberta-base", add_prefix_space=True)
    sentence = ["The", "screen", "is", "great", "but", "battery", "is", "bad", "."]

    # Two targets in the same sentence
    inputs1 = build_v2_inputs(tokenizer, sentence, ["screen"], 1, 2, [1, 0, 1, 2, 3, 4, 5, 6, 7])
    inputs2 = build_v2_inputs(tokenizer, sentence, ["battery"], 5, 6, [4, 3, 2, 1, 1, 0, 1, 2, 3])

    input_ids = torch.cat([inputs1["input_ids"], inputs2["input_ids"]], dim=0)
    attention_mask = torch.cat([inputs1["attention_mask"], inputs2["attention_mask"]], dim=0)
    aspect_mask = torch.cat([inputs1["aspect_mask"], inputs2["aspect_mask"]], dim=0)
    distance_ids = torch.cat([inputs1["distance_ids"], inputs2["distance_ids"]], dim=0)
    evidence_mask = torch.cat([inputs1["evidence_mask"], inputs2["evidence_mask"]], dim=0)
    labels = torch.tensor([2, 0], dtype=torch.long)

    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        aspect_mask=aspect_mask,
        distance_ids=distance_ids,
        evidence_mask=evidence_mask,
        labels=labels,
    )

    assert output.loss is not None
    assert not torch.isnan(output.loss)
    assert not torch.isinf(output.loss)

    # Check backpropagation into parameters
    output.loss.backward()
    assert model.evidence_scorer[1].weight.grad is not None
    assert torch.abs(model.evidence_scorer[1].weight.grad).sum().item() > 0.0

