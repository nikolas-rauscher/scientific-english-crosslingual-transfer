"""
Deterministic Global Order Sampler for exact resume support.
Generates a global deterministic permutation and distributes samples across ranks.
"""
import math
import torch
from torch.utils.data import Sampler
from typing import Iterator
import logging

log = logging.getLogger(__name__)


class DeterministicGlobalSampler(Sampler):
    """
    Deterministic sampler that:
    1. Creates global permutation with fixed seed (reproducible)
    2. Distributes samples across ranks in deterministic order
    3. Supports exact resume from any global sample position
    """
    
    def __init__(
        self,
        dataset_length: int,
        world_size: int,
        rank: int,
        seed: int = 42,
        drop_last: bool = True
    ):
        self.dataset_length = dataset_length
        self.world_size = world_size
        self.rank = rank
        self.seed = seed
        self.drop_last = drop_last
        
        # Generate global deterministic order (same across all ranks)
        generator = torch.Generator()
        generator.manual_seed(seed)
        self.global_order = torch.randperm(dataset_length, generator=generator).tolist()
        
        # Calculate samples per rank
        if drop_last:
            self.samples_per_rank = dataset_length // world_size
            self.total_samples = self.samples_per_rank * world_size
        else:
            self.samples_per_rank = math.ceil(dataset_length / world_size)
            self.total_samples = dataset_length
            
        # Resume support: start position in global order
        self.global_start_position = 0
        
    def set_global_start(self, global_sample_position: int):
        """Set starting position for exact resume."""
        self.global_start_position = global_sample_position % self.dataset_length
        
    def __iter__(self) -> Iterator[int]:
        """Generate indices for this rank starting from global_start_position."""
        indices = []
        global_pos = self.global_start_position
        samples_collected = 0
        samples_checked = 0  # Track how many positions we've checked
        
        while samples_collected < self.samples_per_rank:
            # Get global index from permuted order
            global_idx = self.global_order[global_pos % self.dataset_length]
            
            # Check if this sample belongs to current rank
            if (global_pos % self.world_size) == self.rank:
                indices.append(global_idx)
                samples_collected += 1
                
            global_pos += 1
            samples_checked += 1
            
            # FIXED: Safety check based on samples checked, not global position difference
            # We need to check at most dataset_length * world_size positions to guarantee
            # we find all samples for this rank (worst case: all our samples are at the end)
            if samples_checked > self.dataset_length * self.world_size:
                log.warning(f"Sampler safety break: collected {samples_collected}/{self.samples_per_rank} samples "
                           f"after checking {samples_checked} positions. This may indicate a configuration issue.")
                break
                
        # Diagnostic logging for resume debugging
        if samples_collected < self.samples_per_rank:
            log.warning(f"Sampler returned {samples_collected}/{self.samples_per_rank} samples. "
                       f"This will cause Lightning to end the epoch early!")
        elif self.global_start_position > 0:  # Only log on resume
            log.info(f"Resume successful: collected {samples_collected} samples starting from global position {self.global_start_position}")
                
        return iter(indices)
    
    def __len__(self) -> int:
        return self.samples_per_rank
    
    def get_state(self) -> dict:
        """Get current sampler state for checkpointing."""
        return {
            "global_start_position": self.global_start_position,
            "seed": self.seed,
            "dataset_length": self.dataset_length,
            "world_size": self.world_size,
            "rank": self.rank
        }
    
    def set_state(self, state: dict):
        """Restore sampler state from checkpoint."""
        self.global_start_position = state["global_start_position"]
        # Verify other parameters match
        assert state["seed"] == self.seed
        assert state["dataset_length"] == self.dataset_length
        # Skip world_size and rank checks for flexibility when resuming with different GPU counts
        # assert state["world_size"] == self.world_size
        # assert state["rank"] == self.rank