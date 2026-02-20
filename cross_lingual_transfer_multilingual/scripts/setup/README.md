# Setup Scripts

Scripts for preparing reproducible model artifact layout.

## Recommended order

1. Build layout + lineage manifest:

```bash
/netscratch/anonymous_user/projects/BA-hydra/.venv_pretraining/bin/python \
  cross_lingual_transfer_multilingual/scripts/setup/build_model_layout_and_manifest.py
```

2. Ingest raw artifacts:

- From HF:

```bash
/netscratch/anonymous_user/projects/BA-hydra/.venv_pretraining/bin/python \
  cross_lingual_transfer_multilingual/scripts/setup/download_hf_raw_models.py --languages all
```

- Or from existing local converted checkpoints:

```bash
/netscratch/anonymous_user/projects/BA-hydra/.venv_pretraining/bin/python \
  cross_lingual_transfer_multilingual/scripts/setup/migrate_manifest_sources_to_hf_raw.py --mode symlink
```

3. Canonicalize (`vocab_size == len(tokenizer)`):

```bash
/netscratch/anonymous_user/projects/BA-hydra/.venv_pretraining/bin/python \
  cross_lingual_transfer_multilingual/scripts/setup/canonicalize_models_from_manifest.py
```

## Policy

- No on-the-fly resizing in evaluation.
- Canonicalization is explicit and versioned.
