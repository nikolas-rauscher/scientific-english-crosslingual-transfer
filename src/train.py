from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning as L
import rootutils
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

from src.utils import (
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    if cfg.get("train"):
        log.info("Starting training!")
        
        # Setup datamodule first 
        datamodule.setup()

        # Guard against tokenizer/model vocab mismatch (e.g. T5 extra ids on some HF checkpoints).
        # A mismatch can trigger CUDA device-side asserts during sanity check/validation.
        dm_tokenizer = getattr(datamodule, "tokenizer", None)
        hf_model = getattr(getattr(model, "model", None), "model", None)
        if dm_tokenizer is not None and hf_model is not None:
            try:
                tokenizer_size = int(len(dm_tokenizer))
                embedding_size = int(hf_model.get_input_embeddings().num_embeddings)
                if tokenizer_size > embedding_size:
                    log.warning(
                        "Tokenizer/model vocab mismatch detected: tokenizer=%d, embeddings=%d. "
                        "Resizing token embeddings to %d.",
                        tokenizer_size,
                        embedding_size,
                        tokenizer_size,
                    )
                    hf_model.resize_token_embeddings(tokenizer_size)
                    log.info(
                        "Resized embeddings: new size=%d",
                        int(hf_model.get_input_embeddings().num_embeddings),
                    )
                elif tokenizer_size < embedding_size:
                    log.warning(
                        "Tokenizer/model vocab mismatch detected: tokenizer=%d, embeddings=%d. "
                        "Keeping larger model embeddings unchanged.",
                        tokenizer_size,
                        embedding_size,
                    )
            except Exception as exc:
                log.warning(f"Failed to auto-align tokenizer/model vocab sizes: {exc}")
        
        # Auto-resume support: set sampler position if resuming from checkpoint
        ckpt_path = cfg.get("ckpt_path")
        if ckpt_path:
            import torch
            log.info(f"Resuming from checkpoint: {ckpt_path}")

            # Load checkpoint to get step count
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            completed_steps = int(checkpoint.get("global_step", 0))

            # Use actual trainer settings for world size and accumulation
            num_nodes = getattr(trainer, "num_nodes", 1)
            num_devices = getattr(trainer, "num_devices", 1)
            world_size = max(1, int(num_nodes) * int(num_devices))
            accumulate = max(1, int(getattr(trainer, "accumulate_grad_batches", 1)))
            samples_per_step = int(cfg.data.batch_size) * world_size * accumulate
            total_samples_processed = completed_steps * samples_per_step

            # Hand off to datamodule; the sampler will consume this in train_dataloader()
            if hasattr(datamodule, "set_resume_global_start"):
                datamodule.set_resume_global_start(total_samples_processed)
            else:
                setattr(datamodule, "resume_global_start", total_samples_processed)

            # Coverage logging for operator visibility
            if hasattr(datamodule, 'full_dataset'):
                total_length = datamodule.full_dataset.total_dataset_length
            else:
                total_length = len(datamodule.train_dataset.dataset) if hasattr(datamodule.train_dataset, 'dataset') else len(datamodule.train_dataset)

            coverage_pct = (total_samples_processed / total_length) * 100 if total_length else 0.0
            log.info(
                f"Resume calculation: completed_steps={completed_steps}, "
                f"samples_per_step={samples_per_step}, total_samples_processed={total_samples_processed}"
            )
            log.info(f"Dataset coverage: {coverage_pct:.3f}% ({total_samples_processed}/{total_length})")
            log.info(
                "Will apply resume_global_start to sampler in train_dataloader() "
                f"(start_mod_dataset={total_samples_processed % max(1,total_length)})"
            )
        
        # DEBUG: Check validation interval at runtime
        log.info(f"DEBUG: trainer.val_check_interval = {trainer.val_check_interval}")
        log.info(f"DEBUG: trainer.check_val_every_n_epoch = {trainer.check_val_every_n_epoch}")
        
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    # train the model
    metric_dict, _ = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
