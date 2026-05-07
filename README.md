# Transferring Scientific English Pre-Trained Language Models to Multiple Languages Using Cross-Lingual Transfer

This repository contains the codebase for the paper: **Transferring Scientific English Pre-Trained Language Models to Multiple Languages Using Cross-Lingual Transfer**.

It includes:

- scientific English T5 continued pretraining (EN-T5-Sci)
- multilingual WECHSEL transfer with tokenizers (`spm32k + 100 extra_ids`)
- Global-MMLU evaluation

## Released Models

### Main Models

| Paper name | Language | Hugging Face model |
|---|---|---|
| EN-T5-Sci | English | [rausch/en-t5-sci-continued-pretraining-487k](https://huggingface.co/rausch/en-t5-sci-continued-pretraining-487k) |
| DE-Trans-Init | German | [rausch/de-t5-sci-transfer-init-spm32k](https://huggingface.co/rausch/de-t5-sci-transfer-init-spm32k) |
| JA-Trans-Init | Japanese | [rausch/ja-t5-sci-transfer-init-spm32k](https://huggingface.co/rausch/ja-t5-sci-transfer-init-spm32k) |
| RU-Trans-Init | Russian | [rausch/ru-t5-sci-transfer-init-spm32k](https://huggingface.co/rausch/ru-t5-sci-transfer-init-spm32k) |
| PL-Trans-Init | Polish | [rausch/pl-t5-sci-transfer-init-spm32k](https://huggingface.co/rausch/pl-t5-sci-transfer-init-spm32k) |
| ES-Trans-Init | Spanish | [rausch/es-t5-sci-transfer-init-spm32k](https://huggingface.co/rausch/es-t5-sci-transfer-init-spm32k) |
| PT-Trans-Init | Portuguese | [rausch/pt-t5-sci-transfer-init-spm32k](https://huggingface.co/rausch/pt-t5-sci-transfer-init-spm32k) |

### Continued-Pretraining Control Models

| Paper name | Language | Hugging Face model |
|---|---|---|
| DE-Base-CP | German | [rausch/de-t5-base-sci-cp-15k](https://huggingface.co/rausch/de-t5-base-sci-cp-15k) |
| JA-Base-CP | Japanese | [rausch/ja-t5-base-sci-cp-15k](https://huggingface.co/rausch/ja-t5-base-sci-cp-15k) |
| RU-Base-CP | Russian | [rausch/ru-t5-base-sci-cp-15k](https://huggingface.co/rausch/ru-t5-base-sci-cp-15k) |
| PL-Base-CP | Polish | [rausch/pl-t5-base-sci-cp-15k](https://huggingface.co/rausch/pl-t5-base-sci-cp-15k) |
| ES-Base-CP | Spanish | [rausch/es-t5-base-sci-cp-15k](https://huggingface.co/rausch/es-t5-base-sci-cp-15k) |
| PT-Base-CP | Portuguese | [rausch/pt-t5-base-sci-cp-15k](https://huggingface.co/rausch/pt-t5-base-sci-cp-15k) |

## Project

- cleaning / preparation pipelines: `src/dataprep/**`
- training: `src/train.py`
- evaluation: `src/eval.py`, `src/eval_pipeline.py`
- multilingual pipeline: `cross_lingual_transfer_multilingual/**`
- experiment configs: `configs/experiment/`

## Reproduction

1. Cleaning

```bash
python src/dataprep/pipelines/clean_data.py
```

2. Sliding-window preparation

```bash
python src/dataprep/pipelines/run_sliding_windows.py
```

3. EN continued pretraining

```bash
python src/train.py experiment=t5_continued_pretraining_lr_001_OPTIMIZED_clean_restart
```

4. Multilingual paper pipeline

```bash
bash cross_lingual_transfer_multilingual/jobs/paper/run_paper_pipeline_all.sh
```

## Slurm Job Scripts

For cluster execution:

- `jobs/paper/run_en_cleaning.sbatch`
- `jobs/paper/run_en_sliding_windows.sbatch`
- `jobs/paper/run_en_stats.sbatch`
- `jobs/paper/run_en_spacy_stats.sbatch`
- `jobs/paper/run_en_fasttext_stats.sbatch`
- `jobs/paper/run_en_continued_pretraining.sbatch`
- `jobs/paper/run_submit_dataprep_pipeline.sh` (submits cleaning -> analysis/windows jobs with dependencies)

## Paper Configs

Configs in `configs/experiment/`:

- `t5_continued_pretraining_lr_001_OPTIMIZED_clean_restart.yaml`
- `scientific_crosslingual_transfer_eval_full_15k.yaml`
- `scientific_crosslingual_transfer_eval_full_15k_scifive_en.yaml`
- `clt_multi_mmlu_paper_*.yaml`
- `bloom_original_global_mmlu_{en,ja,es,ru,pl,pt}.yaml`

## Data and Evaluation

- Training corpus: [SciLaD all-text v1](https://huggingface.co/datasets/scilons/SciLaD-all-text-v1)
- Evaluation benchmark: [Global-MMLU](https://huggingface.co/datasets/CohereLabs/Global-MMLU)

## Citation

- Title: Transferring Scientific English Pre-Trained Language Models to Multiple Languages Using Cross-Lingual Transfer
- Authors: Nikolas Rauscher, Fabio Barth, Georg Rehm
- Venue: LREC-COLING 2026, citation details TBA after publication
