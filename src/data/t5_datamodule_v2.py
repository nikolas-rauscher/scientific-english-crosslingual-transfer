"""
T5 DataModule with backwards compatibility support.
Can use either legacy (1.5x heuristic) or optimized (exact T5 formula) collator.
"""

from typing import Optional, Tuple
from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from lightning import LightningDataModule
from lightning.fabric.utilities.types import _Stateful
from transformers import AutoTokenizer
from src.data.components.optimized_t5_collator_v2 import (
    DataCollatorForT5MLM, 
    compute_t5_input_and_target_lengths
)
from src.data.components.t5_dataset import T5MaterializedWindowDataset
from src.data.components.deterministic_sampler import DeterministicGlobalSampler
import os
import logging

log = logging.getLogger(__name__)


class StatefulDataLoader(DataLoader, _Stateful):
    """DataLoader that implements Lightning's _Stateful interface for resumability."""
    
    def state_dict(self):
        """Return state for checkpoint saving."""
        if hasattr(self.sampler, 'get_state'):
            return {'sampler_state': self.sampler.get_state()}
        return {}
    
    def load_state_dict(self, state_dict):
        """Load state from checkpoint."""
        if 'sampler_state' in state_dict and hasattr(self.sampler, 'set_state'):
            self.sampler.set_state(state_dict['sampler_state'])


class T5DataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        tokenizer_name_or_path: str = "t5-base",
        batch_size: int = 8,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        prefetch_factor: int = 4,
        train_val_split: Tuple[float, float] = (0.95, 0.05),
        max_length: int = 512,
        corruption_rate: float = 0.15,
        mean_span_length: int = 3,
        shuffle_buffer_size: int = 10_000,
        use_materialized_windows: bool = True,
        limit_files: int = -1,
        limit_documents: int = -1,
        seed: int = 42,
        use_legacy_collator: bool = True,  # Default to legacy (1.5x) for backwards compatibility
        use_optimized_collator: bool = False,  # Explicitly opt-in to T5 formula mode
        strict_tokenizer_path: bool = True,  # Fail if loaded tokenizer source differs from requested source
        allow_slow_tokenizer_fallback: bool = False,  # If False, fail fast-tokenizer load errors immediately
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)
        
        # Prefer fast tokenizer.
        # Optional fallback to slow tokenizer can be disabled for strict runs.
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name_or_path,
                use_fast=True,
                model_max_length=max_length,
            )
        except Exception as exc:
            if not allow_slow_tokenizer_fallback:
                raise RuntimeError(
                    f"Fast tokenizer load failed for '{tokenizer_name_or_path}' "
                    f"and allow_slow_tokenizer_fallback=False. Error: {exc}"
                ) from exc
            msg = str(exc)
            if "PyPreTokenizerTypeWrapper" not in msg:
                raise
            log.warning(
                "Fast tokenizer load failed for '%s' (%s). Falling back to slow tokenizer.",
                tokenizer_name_or_path,
                exc,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name_or_path,
                use_fast=False,
                model_max_length=max_length,
            )
        self.tokenizer.model_max_length = max_length
        loaded_name_or_path = str(getattr(self.tokenizer, "name_or_path", ""))
        log.info(
            "Tokenizer requested='%s' loaded='%s' class=%s use_fast=%s",
            tokenizer_name_or_path,
            loaded_name_or_path,
            self.tokenizer.__class__.__name__,
            getattr(self.tokenizer, "is_fast", None),
        )
        if strict_tokenizer_path:
            requested = str(tokenizer_name_or_path)
            requested_path = Path(requested)
            loaded_path = Path(loaded_name_or_path) if loaded_name_or_path else None
            if requested_path.exists():
                if not loaded_path or not loaded_path.exists():
                    raise RuntimeError(
                        "Strict tokenizer path check failed: loaded tokenizer "
                        f"source is not a local path. requested='{requested}', "
                        f"loaded='{loaded_name_or_path}'"
                    )
                if requested_path.resolve() != loaded_path.resolve():
                    raise RuntimeError(
                        "Strict tokenizer path check failed: "
                        f"requested='{requested_path.resolve()}', "
                        f"loaded='{loaded_path.resolve()}'"
                    )
            elif loaded_name_or_path != requested:
                raise RuntimeError(
                    "Strict tokenizer path check failed: "
                    f"requested='{requested}', loaded='{loaded_name_or_path}'"
                )

        self.sampler = None
        self.full_dataset = None
        self.train_dataset = None
        self.val_dataset = None
        self.collator = None
        
        # Resume controls (same as V1)
        self.resume_global_start = None  # manual resume fallback (from global_step)
        self._pending_sampler_state = None  # sampler state loaded before sampler exists
        self.train_sampler = None  # Store sampler reference

    def setup(self, stage: Optional[str] = None):
        """Load and split dataset; create optimized collator."""
        if self.train_dataset is not None:
            return  # Already setup
        
        log.info(f"[rank: {os.environ.get('LOCAL_RANK', 0)}] Using materialized windows from: {self.hparams.data_dir}")
        
        # Get parquet files from data directory
        import glob
        parquet_files = glob.glob(f"{self.hparams.data_dir}/*.parquet")
        if not parquet_files:
            raise ValueError(f"No parquet files found in {self.hparams.data_dir}")
        
        # Apply file limit if specified
        if self.hparams.limit_files > 0:
            parquet_files = parquet_files[:self.hparams.limit_files]
            
        full_dataset = T5MaterializedWindowDataset(
            parquet_files=parquet_files,
            limit_documents=self.hparams.limit_documents,
        )
        
        self.full_dataset = full_dataset

        # Compute split lengths
        train_ratio, val_ratio = self.hparams.train_val_split
        train_len = int(len(full_dataset) * train_ratio)
        val_len = len(full_dataset) - train_len
        self.train_dataset, self.val_dataset = random_split(
            full_dataset, [train_len, val_len], generator=torch.Generator().manual_seed(42)
        )

        # Create collator based on mode (check optimized flag first)
        if self.hparams.use_optimized_collator:
            # Optimized mode: calculate correct target_length
            tokens_length, target_length = compute_t5_input_and_target_lengths(
                self.hparams.max_length,
                self.hparams.corruption_rate,
                self.hparams.mean_span_length
            )
            log.info("="*60)
            log.info("[OPTIMIZED MODE] Using exact T5 formula for collator")
            log.info(f"  Input length: {self.hparams.max_length}")
            log.info(f"  Expanded length: {tokens_length} ({tokens_length/self.hparams.max_length:.3f}x)")
            log.info(f"  Target length: {target_length}")
            log.info(f"  Noise density: {self.hparams.corruption_rate}")
            log.info(f"  Mean span length: {self.hparams.mean_span_length}")
            log.info("="*60)
            
            # Set tokenizer max_length to avoid warnings
            self.tokenizer.model_max_length = tokens_length
            use_legacy = False
        else:
            # Legacy mode (default): use 1.5x heuristic with target_length=512
            log.info("="*60)
            log.info("[LEGACY MODE] Using 1.5x heuristic collator (backwards compatibility)")
            log.info(f"  Input length: {self.hparams.max_length}")
            log.info(f"  Expanded length: {int(self.hparams.max_length * 1.5)} (1.5x)")
            log.info(f"  Target length: {self.hparams.max_length} (forced)")
            log.info("  NOTE: This mode is for backwards compatibility with running experiments")
            log.info("="*60)
            target_length = self.hparams.max_length  # 512
            use_legacy = True
        
        self.collator = DataCollatorForT5MLM(
            tokenizer=self.tokenizer,
            noise_density=self.hparams.corruption_rate,
            mean_noise_span_length=self.hparams.mean_span_length,
            input_length=self.hparams.max_length,
            target_length=target_length,
            pad_token_id=self.tokenizer.pad_token_id,
            decoder_start_token_id=self.tokenizer.pad_token_id,
            use_legacy_heuristic=use_legacy,  # Pass the computed flag
        )

    def set_resume_global_start(self, pos: int):
        """Set the global start position for exact resume (applied in train_dataloader)."""
        try:
            self.resume_global_start = int(pos)
        except Exception:
            self.resume_global_start = pos


    # Dataloaders
    def train_dataloader(self):
        # Import here to avoid circular imports
        import lightning as L
        
        # Get distributed info - handle both distributed and single GPU cases
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
        else:
            # Single GPU or not yet initialized
            world_size = 1
            rank = 0
            
        # CRITICAL: Use DeterministicGlobalSampler for exact resume support  
        # IMPORTANT: Must match the dataset we actually iterate over (train_dataset)
        sampler = DeterministicGlobalSampler(
            dataset_length=len(self.train_dataset),  # Must match DataLoader dataset
            world_size=world_size,
            rank=rank,
            seed=self.hparams.seed,
            drop_last=True
        )
        # Store sampler reference for resume support
        self.train_sampler = sampler
        
        # Apply any checkpoint-loaded sampler state (preferred), else manual resume offset
        from src.utils.pylogger import RankedLogger
        log = RankedLogger(__name__, rank_zero_only=True)
        # Prefer precise resume from global_step calculation when available.
        if self.resume_global_start is not None:
            sampler.set_global_start(self.resume_global_start)
            log.info(f"Applying resume_global_start={self.resume_global_start} (preferred over sampler_state)")
        elif getattr(self, "_pending_sampler_state", None) is not None:
            try:
                sampler.set_state(self._pending_sampler_state)
                log.info("Applied sampler state from checkpoint into fresh sampler (no global_step fallback available)")
            except Exception:
                log.warning("Failed to apply sampler_state; starting from position 0")
        
        # DEBUG: Log dataset lengths for epoch calculation verification
        log.info(f"Dataset length verification:")
        log.info(f"  - Full dataset length: {len(self.full_dataset):,}")
        log.info(f"  - Train split length: {len(self.train_dataset):,}")
        log.info(f"  - Sampler dataset_length (used): {sampler.dataset_length:,}")
        log.info(f"  - Sampler samples_per_rank: {sampler.samples_per_rank:,}")
        log.info(f"  - Sampler len(): {len(sampler):,}")
        log.info(f"  - Expected batches per epoch per rank: {len(sampler) // self.hparams.batch_size:,}")
        log.info(f"  - Expected optimizer steps per epoch: {(len(sampler) // self.hparams.batch_size) // 2:,} (with accumulate=2)")
        
        dataloader = StatefulDataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            sampler=sampler,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.collator,
            prefetch_factor=self.hparams.prefetch_factor if self.hparams.num_workers > 0 else None,
            persistent_workers=self.hparams.persistent_workers if self.hparams.num_workers > 0 else False,
            drop_last=True,
        )
        
        return dataloader


    def val_dataloader(self):
        # Simple validation without distributed sampler (same as V1)
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers,
            prefetch_factor=self.hparams.prefetch_factor,
            shuffle=False,
            collate_fn=self.collator,
            drop_last=False,  # Keep all validation data
        )
            
    def on_after_batch_transfer(self, batch, dataloader_idx):
        """Apply label masking after batch is on correct device."""
        # Move label padding to -100 if not already done
        if "labels" in batch:
            # Make sure padding tokens are set to -100 for loss computation
            batch["labels"][batch["labels"] == self.tokenizer.pad_token_id] = -100
        return batch

    def state_dict(self) -> dict:
        """Get datamodule state for Lightning resumability."""
        state = {}
        if hasattr(self, 'train_sampler'):
            state['sampler_state'] = self.train_sampler.get_state()
        return state
    
    def load_state_dict(self, state_dict: dict) -> None:
        """Load datamodule state for Lightning resumability.
        If sampler isn't constructed yet, stash state and apply in train_dataloader().
        """
        if 'sampler_state' in state_dict:
            if hasattr(self, 'train_sampler') and self.train_sampler is not None:
                self.train_sampler.set_state(state_dict['sampler_state'])
            else:
                self._pending_sampler_state = state_dict['sampler_state']
