# Cross-Lingual Transfer Multilingual (Paper Track)

Paper-only multilingual pipeline for:

- `deu_Latn`, `jpn_Jpan`, `spa_Latn`, `rus_Cyrl`, `pol_Latn`, `por_Latn`
- target tokenizer training (`spm32k + 100 extra_ids`)
- WECHSEL initialization
- 15k continued pretraining
- Global-MMLU evaluation + aggregation

## Canonical Orchestrator

```bash
bash cross_lingual_transfer_multilingual/jobs/paper/run_paper_pipeline_all.sh
```

This orchestrates:

1. `run_01_tokenizers_all.sbatch`
2. `run_02_wechsel_init_all.sbatch`
3. `run_03_submit_continued_all.sh`
4. `run_04_prepare_permanent_all.sbatch`
5. `run_05_submit_mmlu_all.sh`
6. `run_06_aggregate_all.sbatch`

## Key Scripts

- `scripts/paper/train_target_tokenizers_all.py`
- `scripts/paper/run_wechsel_transfer_all.py`
- `scripts/paper/run_continued_pretraining_single_lang_paper.sh`
- `scripts/paper/prepare_permanent_models_paper.py`
- `scripts/paper/generate_mmlu_configs_all.py`
- `scripts/paper/aggregate_paper_ab_results.py`

## Status Source of Truth

- `cross_lingual_transfer_multilingual/docs/CURRENT_STATUS.md`
- `cross_lingual_transfer_multilingual/docs/PAPER_TRACK_SESSION_HANDOVER_2026-02-17.md`
