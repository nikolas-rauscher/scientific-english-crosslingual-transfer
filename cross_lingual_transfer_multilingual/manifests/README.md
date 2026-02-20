# manifests

Lineage and normalization metadata for multilingual model artifacts.

## Files

- `model_lineage_v1.json`: source/current paths and planned `hf_raw`, `canonical_v1`, `eval_variant_v1` locations.

## Workflow

1. Build/refresh lineage scaffold.
2. Migrate/download raw artifacts.
3. Canonicalize raw artifacts.
4. Run strict evaluation from canonical artifacts.
