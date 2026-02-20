help: ## Show commands
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-28s\033[0m %s\n", $$1, $$2}'

clean-pyc: ## Remove Python cache files
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type f -name "*.py[co]" -delete

paper-cleaning: ## Run EN cleaning pipeline
	python src/dataprep/pipelines/clean_data.py

paper-windows: ## Run EN sliding-window preparation
	python src/dataprep/pipelines/run_sliding_windows.py

paper-train-en: ## Run EN continued pretraining experiment
	python src/train.py experiment=t5_continued_pretraining_lr_001_OPTIMIZED_clean_restart

paper-pipeline: ## Run multilingual paper pipeline (all stages)
	bash cross_lingual_transfer_multilingual/jobs/paper/run_paper_pipeline_all.sh

paper-mmlu: ## Submit paper multilingual Global-MMLU stage
	bash cross_lingual_transfer_multilingual/jobs/paper/run_05_submit_mmlu_all.sh

paper-aggregate: ## Run paper aggregation stage
	sbatch cross_lingual_transfer_multilingual/jobs/paper/run_06_aggregate_all.sbatch

paper-jobs-dataprep: ## Submit EN dataprep pipeline jobs (cleaning + stats + windows)
	bash jobs/paper/run_submit_dataprep_pipeline.sh

paper-job-train-en: ## Submit EN continued pretraining via sbatch wrapper
	sbatch jobs/paper/run_en_continued_pretraining.sbatch
