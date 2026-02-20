# Transferring Scientific English Pre-Trained Language Models to Multiple Languages Using Cross-Lingual Transfer

This repository is codebase for: Transferring Scientific English Pre-Trained Language Models to Multiple Languages Using Cross-Lingual Transfer

- scientific English T5 continued pretraining (EN-T5-Sci)
- multilingual WECHSEL transfer with paper tokenizers (`spm32k + 100 extra_ids`)
- Global-MMLU evaluation and paper aggregation

## Project

- cleaning / preparation pipelines: `src/dataprep/**`
- training: `src/train.py`
- evaluation: `src/eval.py`, `src/eval_pipeline.py`
- multilingual pipeline: `cross_lingual_transfer_multilingual/**`
- experiments in `configs/experiment/`



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

