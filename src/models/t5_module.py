from __future__ import annotations

"""
pytorch lightning module for t5 pre-training with span corruption.

this module is kept simple, offloading most work to hugging face's implementation.
it primarily handles:

logging metrics (loss + perplexity)
setting up optimizers and schedulers via hydra
leveraging lightning's support for mixed precision and distributed training.
"""

from typing import Any, Dict
import warnings
import time

import torch
from lightning import LightningModule
from torch.optim.optimizer import Optimizer
from torchmetrics import MeanMetric

from transformers import get_cosine_with_hard_restarts_schedule_with_warmup

from .components.t5_model import T5Model

from src.utils.pylogger import RankedLogger

try:
    import wandb
except ImportError:
    wandb = None

# Initialize logger
logger = RankedLogger(__name__, rank_zero_only=True)


class T5LitModule(LightningModule):
    """Lightning wrapper around :class:`~src.models.components.t5_model.T5Model`."""

    def __init__(
        self,
        t5_model: T5Model,
        optimizer: torch.optim.Optimizer,
        scheduler: Dict[str, Any] | None = None,
        tokenizer=None,  # Add tokenizer for span corruption logging
        ignore_index: int = -100,  # Configurable ignore index for perplexity
        label_smoothing: float = 0.0,  # Optional label smoothing for stability
    ) -> None:  # noqa: D401
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.model = t5_model
        self.tokenizer = tokenizer

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        
        # Initialize Perplexity metrics with configurable ignore_index
        # Use tokenizer pad_token_id if available, otherwise use configured ignore_index
        final_ignore_index = ignore_index
        if tokenizer is not None and hasattr(tokenizer, 'pad_token_id') and tokenizer.pad_token_id is not None:
            final_ignore_index = tokenizer.pad_token_id
        
        # Use efficient exp(loss) perplexity calculation for both train and validation
        
        # Token throughput tracking
        self.step_start_time = None
        self.total_tokens_processed = 0
        self.throughput_window_tokens = 0
        self.throughput_window_start = None
        self.throughput_window_size = 100  # Calculate throughput over N steps
        self.last_step_end_time = None
        
        # Training run estimation
        self.training_start_time = None
        self.eta_calculation_interval = 50  # Calculate ETA every N steps
        self.estimated_total_steps = None  # Will be estimated from dataset size

    def forward(self, **batch):  # type: ignore[override]
        return self.model(**batch)

    def _compute_smoothed_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute CrossEntropy with optional label smoothing and ignore_index."""
        ignore_index = getattr(self.hparams, "ignore_index", -100)
        ls = float(getattr(self.hparams, "label_smoothing", 0.0) or 0.0)
        vocab_size = logits.size(-1)
        logits = logits.view(-1, vocab_size)
        labels = labels.view(-1)

        if ls <= 0.0:
            import torch.nn.functional as F
            return F.cross_entropy(logits, labels, ignore_index=ignore_index)

        try:
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, label_smoothing=ls)
            return loss_fct(logits, labels)
        except TypeError:
            # Fallback for older torch without label_smoothing in CrossEntropyLoss
            import torch.nn.functional as F
            log_probs = F.log_softmax(logits, dim=-1)
            # Mask out ignore indices
            active = labels != ignore_index
            if active.sum() == 0:
                return torch.zeros((), device=logits.device, dtype=logits.dtype)
            labels_active = labels[active]
            log_probs_active = log_probs[active]
            nll_loss = F.nll_loss(log_probs_active, labels_active, reduction='mean')
            smooth_loss = -log_probs_active.mean(dim=-1).mean()
            return (1.0 - ls) * nll_loss + ls * smooth_loss

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):  # type: ignore[override]
        # DDP Debug: Log unique doc_ids per rank (only first few steps)  
        if self.global_step < 5 and batch_idx == 0:
            # Extract doc_ids from input_ids for debugging (simplified check)
            sample_tokens = batch.get('input_ids', torch.tensor([]))[0][:5].tolist() if len(batch.get('input_ids', [])) > 0 else []
            logger.info(f"[RANK {self.global_rank}] Step {self.global_step} sample_tokens: {sample_tokens}")
        
        outputs = self.forward(**batch)
        # Use manual loss with label smoothing when configured
        if getattr(self.hparams, "label_smoothing", 0.0) and "labels" in batch:
            loss = self._compute_smoothed_loss(outputs.logits, batch["labels"])
        else:
            loss = outputs.loss

        self.train_loss(loss)
        
        # Calculate perplexity efficiently using exp(loss) instead of expensive softmax
        # Note: For language modeling, perplexity = exp(cross_entropy_loss)
        train_perplexity = torch.exp(loss.detach())  # Detach to avoid grad graph issues
        
        # Log loss per step AND per epoch
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/loss_epoch", self.train_loss, on_step=False, on_epoch=True, prog_bar=False)
        
        # Log perplexity - use efficient exp(loss) calculation  
        self.log("train/perplexity", train_perplexity, on_step=True, on_epoch=True, prog_bar=True)
        
        current_lr = self.optimizers().param_groups[0]['lr']
        if current_lr is not None:
            self.log("train/learning_rate", current_lr, on_step=True, prog_bar=True)
        
        batch_size = batch["input_ids"].size(0)
        attention_mask = batch["attention_mask"]
        real_tokens = attention_mask.sum().item()
        total_tokens = attention_mask.numel()
        padding_ratio = 1 - (real_tokens / total_tokens)
        
        # Update cumulative tracking
        self.total_tokens_processed += real_tokens
        
        # Get GPU count for global throughput calculation
        num_gpus = self.trainer.num_devices if hasattr(self.trainer, 'num_devices') else 1
        effective_batch_size = batch_size * num_gpus * getattr(self.trainer, 'accumulate_grad_batches', 1)
        global_tokens_per_step = real_tokens * num_gpus
        
        self.log("train/batch_size", batch_size, on_step=True)
        self.log("train/effective_batch_size", effective_batch_size, on_step=True)
        self.log("train/num_gpus", num_gpus, on_step=True)
        self.log("train/padding_ratio", padding_ratio, on_step=True)
        self.log("train/tokens_per_batch", real_tokens, on_step=True)
        self.log("train/global_tokens_per_step", global_tokens_per_step, on_step=True)
        self.log("train/total_tokens_processed", self.total_tokens_processed, on_step=True)
        
        if batch_idx % 100 == 0 and self.tokenizer is not None:
            self._log_span_corruption_examples(batch, outputs)
        
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):  # type: ignore[override]
        outputs = self.forward(**batch)
        if getattr(self.hparams, "label_smoothing", 0.0) and "labels" in batch:
            loss = self._compute_smoothed_loss(outputs.logits, batch["labels"])
        else:
            loss = outputs.loss

        self.val_loss(loss)
        
        # Debug: Check for NaN loss
        if torch.isnan(loss):
            print(f"WARNING: NaN loss detected at validation step {batch_idx}")
        
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        
        # Calculate perplexity efficiently using exp(loss) - same as training
        val_perplexity = torch.exp(self.val_loss.compute().detach())
        self.log("val/perplexity", val_perplexity, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            
        return loss

    def configure_optimizers(self):  # type: ignore[override]
        # Optimizer from Hydra config
        optimizer_cfg = self.hparams.optimizer
        if isinstance(optimizer_cfg, Optimizer):
            optimizer = optimizer_cfg
        elif hasattr(optimizer_cfg, 'func'):  # Hydra partial function
            # Filter out null/None parameters before passing to optimizer
            filtered_params = {}
            for key, value in optimizer_cfg.keywords.items():
                if value is not None:
                    filtered_params[key] = value
            
            # Create a new partial with filtered parameters
            from functools import partial
            filtered_optimizer_cfg = partial(optimizer_cfg.func, **filtered_params)
            optimizer = filtered_optimizer_cfg(self.parameters())
        else:
            # Fallback
            optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)

        # No scheduler case
        if self.hparams.scheduler is None:
            return optimizer

        # Scheduler from Hydra config
        scheduler_cfg = self.hparams.scheduler
        
        # Modern Hydra way - scheduler is already instantiated as partial
        if hasattr(scheduler_cfg, 'func'):  # Check if it's a partial function
            scheduler = scheduler_cfg(optimizer)
        elif hasattr(scheduler_cfg, 'get'):  # Check if it's a dict
            # Legacy fallback for dict config
            scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
                optimizer,
                num_warmup_steps=scheduler_cfg.get("num_warmup_steps", 1000),
                num_training_steps=scheduler_cfg.get("num_training_steps", 100_000),
                num_cycles=scheduler_cfg.get("num_cycles", 1),
            )
        else:
            # Fallback: assume it's a callable that takes optimizer
            scheduler = scheduler_cfg(optimizer)
            
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
    
    def on_before_optimizer_step(self, optimizer):
        total_norm = 0
        param_count = 0
        
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
        
        total_norm = total_norm ** (1. / 2) if total_norm > 0 else 0.0
        self.log("train/grad_norm", total_norm, on_step=True)
        self.log("train/trainable_params", param_count, on_step=True)
    
    def _log_span_corruption_examples(self, batch: Dict[str, torch.Tensor], outputs):
        if wandb is None or not hasattr(self.logger, 'experiment'):
            return
            
        try:
            input_ids = batch["input_ids"][0]
            labels = batch["labels"][0]
            
            input_text = self.tokenizer.decode(input_ids, skip_special_tokens=False)
            target_text = self.tokenizer.decode(labels, skip_special_tokens=False)
            
            with torch.no_grad():
                pred_ids = self.model.generate(
                    input_ids.unsqueeze(0),
                    max_length=50,
                    do_sample=False,
                    num_beams=1
                )
            pred_text = self.tokenizer.decode(pred_ids[0], skip_special_tokens=False)
            
            example_table = wandb.Table(
                columns=["Step", "Input", "Target", "Prediction"],
                data=[[self.global_step, input_text[:200], target_text[:200], pred_text[:200]]]
            )
            
            self.logger.experiment.log({
                "span_corruption_examples": example_table,
                "global_step": self.global_step
            })
            
        except Exception as e:
            pass 
    
    def on_train_batch_start(self, batch, batch_idx):
        """Called at the beginning of each training batch."""
        self.step_start_time = time.time()
        
        # Initialize training start time on first batch
        if self.training_start_time is None:
            self.training_start_time = self.step_start_time
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Called at the end of each training batch (after optimizer step)."""
        if self.step_start_time is None:
            return
            
        step_end_time = time.time()
        step_duration = step_end_time - self.step_start_time
        
        # Track recent step times for better ETA calculation (last 100 steps)
        if not hasattr(self, 'recent_step_times'):
            self.recent_step_times = []
        self.recent_step_times.append(step_duration)
        if len(self.recent_step_times) > 100:
            self.recent_step_times.pop(0)
        
        # Calculate tokens for this batch (across all GPUs)
        attention_mask = batch["attention_mask"]
        real_tokens = attention_mask.sum().item()
        num_gpus = self.trainer.num_devices if hasattr(self.trainer, 'num_devices') else 1
        global_tokens = real_tokens * num_gpus
        
        # Real throughput: includes forward, backward, optimizer step, synchronization
        tokens_per_second = global_tokens / step_duration if step_duration > 0 else 0
        
        # Window-based average calculation
        if self.throughput_window_start is None:
            self.throughput_window_start = self.step_start_time
            self.throughput_window_tokens = 0
        
        self.throughput_window_tokens += global_tokens
        
        # Log instantaneous throughput (live in progress bar)
        self.log("train/tokens_per_second_real", tokens_per_second, on_step=True, prog_bar=True)
        self.log("train/step_duration_ms", step_duration * 1000, on_step=True, prog_bar=True)
        
        # Calculate and log rolling average every N steps
        if batch_idx % self.throughput_window_size == 0 and batch_idx > 0:
            window_duration = step_end_time - self.throughput_window_start
            avg_tokens_per_second = self.throughput_window_tokens / window_duration if window_duration > 0 else 0
            
            # Reset window
            self.throughput_window_start = step_end_time
            self.throughput_window_tokens = 0
            
            self.log("train/tokens_per_second_avg", avg_tokens_per_second, on_step=True, prog_bar=True)
        
        # Calculate ETA and training projections (every 10 steps for more responsive updates)
        if batch_idx % 10 == 0 and batch_idx > 0:
            self._calculate_training_eta(batch_idx, step_end_time, global_tokens)
        
        self.step_start_time = None
    
    def _calculate_training_eta(self, current_step, current_time, tokens_this_step):
        """Calculate estimated time to completion for the training run."""
        if self.training_start_time is None:
            return
            
        # Get training configuration
        max_steps = getattr(self.trainer, 'max_steps', None)
        max_epochs = getattr(self.trainer, 'max_epochs', None)
        
        # Estimate dataset size if not done yet
        if self.estimated_total_steps is None:
            self._estimate_dataset_size()
        
        # Determine the actual training limit (steps vs dataset vs epochs)
        actual_max_steps = self._get_actual_training_limit(max_steps, max_epochs)
        
        if actual_max_steps is None or actual_max_steps <= 0:
            # No training limit set - log a message and skip ETA calculation
            self.log("eta/status", "unlimited_training", on_step=True)
            return
            
        # Calculate elapsed time and progress
        elapsed_time = current_time - self.training_start_time
        elapsed_hours = elapsed_time / 3600
        
        # Progress calculations
        steps_completed = self.global_step
        steps_remaining = actual_max_steps - steps_completed
        progress_pct = (steps_completed / actual_max_steps) * 100 if actual_max_steps > 0 else 0
        
        # Time estimates
        if steps_completed > 0:
            # Use recent step timing instead of total elapsed time for more accurate ETA
            if hasattr(self, 'recent_step_times') and len(self.recent_step_times) > 0:
                avg_time_per_step = sum(self.recent_step_times) / len(self.recent_step_times)
            else:
                avg_time_per_step = elapsed_time / steps_completed
            eta_seconds = avg_time_per_step * steps_remaining
            eta_hours = eta_seconds / 3600
            eta_days = eta_hours / 24
            
            # Token estimates
            total_tokens_target = self.total_tokens_processed * (actual_max_steps / steps_completed) if steps_completed > 0 else 0
            tokens_remaining = total_tokens_target - self.total_tokens_processed
            
            # Current throughput
            current_tokens_per_hour = self.total_tokens_processed / elapsed_hours if elapsed_hours > 0 else 0
            
            # Core ETA metrics (simplified)
            self.log("eta/progress_percent", progress_pct, on_step=True, prog_bar=True)
            self.log("eta/remaining_days", eta_days, on_step=True, prog_bar=True)
            self.log("eta/remaining_hours", eta_hours, on_step=True)
            self.log("eta/steps_remaining", steps_remaining, on_step=True)
            self.log("eta/total_steps", actual_max_steps, on_step=True)
            
            # Performance metrics
            self.log("eta/tokens_per_hour", current_tokens_per_hour, on_step=True)
            steps_per_hour = 3600 / avg_time_per_step if avg_time_per_step > 0 else 0
            self.log("eta/steps_per_hour", steps_per_hour, on_step=True)
    
    def _estimate_dataset_size(self):
        """Estimate total dataset size in steps."""
        try:
            # Try to get dataset size from trainer's datamodule
            if hasattr(self.trainer, 'datamodule') and self.trainer.datamodule is not None:
                datamodule = self.trainer.datamodule
                
                # Check if datamodule has train_dataloader
                if hasattr(datamodule, 'train_dataloader'):
                    train_loader = datamodule.train_dataloader()
                    if hasattr(train_loader, '__len__'):
                        dataset_steps_per_epoch = len(train_loader)
                        max_epochs = getattr(self.trainer, 'max_epochs', 1)
                        self.estimated_total_steps = dataset_steps_per_epoch * max_epochs
                        return
            
            # Fallback: Try to get from trainer's train_dataloader
            if hasattr(self.trainer, 'train_dataloader') and self.trainer.train_dataloader is not None:
                train_loader = self.trainer.train_dataloader
                if hasattr(train_loader, '__len__'):
                    dataset_steps_per_epoch = len(train_loader)
                    max_epochs = getattr(self.trainer, 'max_epochs', 1)
                    self.estimated_total_steps = dataset_steps_per_epoch * max_epochs
                    return
            
            # If we can't determine dataset size, keep it None
            self.estimated_total_steps = None
            
        except Exception:
            # If anything fails, just keep it None
            self.estimated_total_steps = None
    
    def _get_actual_training_limit(self, max_steps, max_epochs):
        """Determine the actual training limit considering steps, epochs, and dataset size."""
        limits = []
        
        # Add configured max_steps if set
        if max_steps is not None and max_steps > 0:
            limits.append(max_steps)
        
        # Add estimated dataset size if available
        if self.estimated_total_steps is not None and self.estimated_total_steps > 0:
            limits.append(self.estimated_total_steps)
        
        # Return the minimum (most restrictive) limit
        if limits:
            return min(limits)
        
        # Fallback: if no limits are set, no ETA calculation possible
        return None
