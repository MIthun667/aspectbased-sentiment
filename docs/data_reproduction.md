# Dataset Reproduction Commands

Run all commands from the repository root.

## 1. Prepare canonical aspect-level records

```bash
python scripts/prepare_data.py \
  --root . \
  --output-root data/processed \
  --strict
```

## 2. Create the protected validation split

```bash
python scripts/create_validation_split.py \
  --processed-root data/processed \
  --validation-ratio 0.15 \
  --seed 2026
```

## 3. Audit canonical splits

```bash
python scripts/audit_canonical_splits.py \
  --processed-root data/processed \
  --output data/audits/canonical_split_audit.json
```

## 4. Report duplicate overlaps

```bash
python scripts/report_split_overlaps.py \
  --processed-root data/processed \
  --output data/audits/split_overlap_details.json
```

## 5. Run tests

```bash
pytest -q
```

Expected result:

```text
27 passed
```

## Expected audit summary

```text
split_errors: 0
partition_errors: 0
sentence_id_overlaps: 0
normalized_text_overlaps: 5
target_signature_overlaps: 4
```

All remaining normalized-text overlaps must occur only between the official
training and test partitions.
