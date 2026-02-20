from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
from lightning.fabric.utilities.types import _Stateful
from .components.deterministic_sampler import DeterministicGlobalSampler
from transformers import (
    PreTrainedTokenizerFast,
    T5TokenizerFast,
)

# Import optimized collator
from .components.optimized_t5_collator import DataCollatorForT5MLM

from .components.t5_dataset import T5ParquetDataset, T5MaterializedWindowDataset


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
        tokenizer_name_or_path: str = "t5-base",  # Use pretrained T5 tokenizer 
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
        use_materialized_windows: bool = True,  # Use materialized windows by default 
        limit_files: int = -1,
        limit_documents: int = -1,  # Limit total number of windows/documents
        seed: int = 42,  # Seed for reproducible shuffling
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.data_dir = Path(data_dir)
        self.tokenizer_name_or_path = tokenizer_name_or_path

        self.train_dataset = None
        self.val_dataset = None

        # Resume controls
        self.resume_global_start = None  # manual resume fallback (from global_step)
        self._pending_sampler_state = None  # sampler state loaded before sampler exists


    # Tokenizer utilities old 
    def prepare_data(self): 
        # skip 
        pass
            
            
    def _load_tokenizer(self) -> PreTrainedTokenizerFast:
        tokenizer = T5TokenizerFast.from_pretrained(self.tokenizer_name_or_path)
        tokenizer.model_max_length = self.hparams.max_length
        return tokenizer
    
    # Setup datasets
    def setup(self, stage: str | None = None): 
        if self.train_dataset is not None:
            return  # Already initialised

        parquet_files = list(self.data_dir.rglob("*.parquet"))
        if not parquet_files:
            raise RuntimeError(f"No Parquet files found in {self.data_dir!s}")

        # Apply file limit if specified
        if self.hparams.limit_files > 0:
            parquet_files = parquet_files[:self.hparams.limit_files]

        # Load tokenizer first (needed for dataset)
        self.tokenizer = self._load_tokenizer()

        from src.utils.pylogger import RankedLogger
        log = RankedLogger(__name__, rank_zero_only=True)
        
        log.info(f"Using materialized windows from: {self.data_dir}")
        
        full_dataset = T5MaterializedWindowDataset(
            parquet_files=parquet_files,
            limit_documents=self.hparams.limit_documents
        )
        
        # Store reference to full dataset for resume support
        self.full_dataset = full_dataset

        # Compute split lengths
        train_ratio, val_ratio = self.hparams.train_val_split
        train_len = int(len(full_dataset) * train_ratio)
        val_len = len(full_dataset) - train_len
        self.train_dataset, self.val_dataset = random_split(
            full_dataset, [train_len, val_len], generator=torch.Generator().manual_seed(42)
        )

        # Create optimized collator with all required parameters
        # Import compute function to calculate correct target_length
        from src.data.components.optimized_t5_collator import compute_t5_input_and_target_lengths
        
        # Calculate correct target_length based on T5 formula
        _, target_length = compute_t5_input_and_target_lengths(
            self.hparams.max_length,
            self.hparams.corruption_rate,
            self.hparams.mean_span_length
        )
        
        self.collator = DataCollatorForT5MLM(
            tokenizer=self.tokenizer,
            noise_density=self.hparams.corruption_rate,
            mean_noise_span_length=self.hparams.mean_span_length,
            input_length=self.hparams.max_length,
            target_length=target_length,  # Use calculated target_length (114 for standard params)
            pad_token_id=self.tokenizer.pad_token_id,
            decoder_start_token_id=self.tokenizer.pad_token_id,  # T5 uses pad_token_id for decoder start
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
        
        return StatefulDataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers,
            prefetch_factor=self.hparams.prefetch_factor,
            sampler=sampler,
            collate_fn=self.collator,
            drop_last=True,
        )

    def val_dataloader(self): 
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers,
            prefetch_factor=self.hparams.prefetch_factor,
            shuffle=False,
            collate_fn=self.collator,
            drop_last=False,          # Keep all validation data
        )

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
