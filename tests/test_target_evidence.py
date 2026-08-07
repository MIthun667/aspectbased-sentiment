from __future__ import annotations

import torch
from src.models.target_evidence import TargetEvidenceModel, TargetEvidenceOutput


def test_target_evidence_model_forward():
    model = TargetEvidenceModel(
        backbone_name="roberta-base",
        num_labels=3,
        max_dependency_distance=10,
    )
    model.eval()

    batch_size = 2
    seq_len = 16

    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)
    aspect_mask = torch.zeros((batch_size, seq_len), dtype=torch.long)
    aspect_mask[:, 2:4] = 1  # tokens 2 and 3 are aspect
    distance_ids = torch.randint(0, 10, (batch_size, seq_len))
    evidence_mask = torch.ones((batch_size, seq_len), dtype=torch.long)
    evidence_mask[:, 0] = 0  # special token <s>
    evidence_mask[:, -1] = 0  # special token </s>
    labels = torch.tensor([0, 2], dtype=torch.long)

    with torch.no_grad():
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            aspect_mask=aspect_mask,
            distance_ids=distance_ids,
            evidence_mask=evidence_mask,
            labels=labels,
        )

    assert output.loss is not None
    assert output.loss.ndim == 0
    assert not torch.isnan(output.loss)
    assert not torch.isinf(output.loss)
    assert output.logits.shape == (batch_size, 3)
    assert output.evidence_attentions is not None
    assert output.evidence_attentions.shape == (batch_size, seq_len)

    # Check evidence attention sums to 1.0 for valid evidence tokens
    attn_sum = output.evidence_attentions.sum(dim=-1)
    assert torch.allclose(attn_sum, torch.ones(batch_size), atol=1e-5)
