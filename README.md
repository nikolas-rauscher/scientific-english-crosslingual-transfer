# BA Hydra Paper Core

This repository is the paper-focused codebase for:

- scientific English T5 continued pretraining (EN-T5-Sci)
- multilingual WECHSEL transfer with paper tokenizers (`spm32k + 100 extra_ids`)
- Global-MMLU evaluation and paper aggregation

Everything outside this scope was pruned on branch `paper-core-refactor-2026-02-20`.

## Project Scope

In scope:

- cleaning / preparation pipelines: `src/dataprep/**`
- training: `src/train.py`
- evaluation: `src/eval.py`, `src/eval_pipeline.py`
- paper multilingual pipeline: `cross_lingual_transfer_multilingual/**`
- paper experiments in `configs/experiment/`
- paper evidence in:
  - `evaluation_results/scientific_crosslingual_transfer_eval_full_15k/`
  - `evaluation/evaluation_results/scientific_crosslingual_transfer_eval_full_15k/`
  - `results/final_evaluation_EN_DE_MMLU_results/scientific_crosslingual_transfer_eval_full_15k/`

## Canonical Commands (5)

1) Cleaning

```bash
python src/dataprep/pipelines/clean_data.py
```

2) Sliding-window preparation

```bash
python src/dataprep/pipelines/run_sliding_windows.py
```

3) EN continued pretraining (paper baseline)

```bash
python src/train.py experiment=t5_continued_pretraining_lr_001_OPTIMIZED_clean_restart
```

4) Multilingual paper pipeline (tokenizer -> WECHSEL -> continued -> prepare)

```bash
bash cross_lingual_transfer_multilingual/jobs/paper/run_paper_pipeline_all.sh
```

5) MMLU + aggregation

```bash
bash cross_lingual_transfer_multilingual/jobs/paper/run_05_submit_mmlu_all.sh
```

## Slurm Job Scripts

For cluster execution, the paper branch now keeps these wrappers:

- `jobs/paper/run_en_cleaning.sbatch`
- `jobs/paper/run_en_sliding_windows.sbatch`
- `jobs/paper/run_en_stats.sbatch`
- `jobs/paper/run_en_spacy_stats.sbatch`
- `jobs/paper/run_en_fasttext_stats.sbatch`
- `jobs/paper/run_en_continued_pretraining.sbatch`
- `jobs/paper/run_submit_dataprep_pipeline.sh` (submits cleaning -> analysis/windows jobs with dependencies)

## Paper Configs

Retained paper configs in `configs/experiment/`:

- `t5_continued_pretraining_lr_001_OPTIMIZED_clean_restart.yaml`
- `scientific_crosslingual_transfer_eval_full_15k.yaml`
- `scientific_crosslingual_transfer_eval_full_15k_scifive_en.yaml`
- `clt_multi_mmlu_paper_*.yaml`
- `bloom_original_global_mmlu_{en,ja,es,ru,pl,pt}.yaml`

## Refactor Docs

- `docs/refactor/KEEP_DROP_MATRIX.md`
- `docs/refactor/PAPER_COMMAND_MAP.md`
