# AGENT.md - Multilingual Paper Track

## Scope

Only paper-track execution is supported in this refactored branch.

## Required flow

1. Train paper tokenizers
2. Build WECHSEL paper init models
3. Run continued pretraining
4. Prepare permanent models
5. Run multilingual Global-MMLU
6. Aggregate results

## Canonical command

```bash
bash cross_lingual_transfer_multilingual/jobs/paper/run_paper_pipeline_all.sh
```

## Follow-up commands

```bash
bash cross_lingual_transfer_multilingual/jobs/paper/run_05_submit_mmlu_all.sh
sbatch cross_lingual_transfer_multilingual/jobs/paper/run_06_aggregate_all.sbatch
```
