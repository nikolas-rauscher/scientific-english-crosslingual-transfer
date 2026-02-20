"""Fast PyTorch-based span masking for T5."""

from typing import List, Tuple
import torch
import numpy as np


def apply_fast_span_corruption(
    tokens: List[int],
    tokenizer,
    corruption_rate: float = 0.15,
    mean_span_length: int = 3,
    sentinel_start_id: int | None = None,
) -> Tuple[List[int], List[int]]:
    """Fast PyTorch-based span corruption for T5."""
    
    if sentinel_start_id is None:
        sentinel_start_id = tokenizer.convert_tokens_to_ids("<extra_id_0>")
    
    # Convert to torch tensor for faster processing
    tokens_tensor = torch.tensor(tokens, dtype=torch.long)
    length = len(tokens)
    
    # Calculate noise parameters
    num_noise_tokens = max(1, min(int(round(length * corruption_rate)), length - 1))
    num_noise_spans = max(1, int(round(num_noise_tokens / mean_span_length)))
    
    # Create noise mask using faster torch operations
    noise_mask = torch.zeros(length, dtype=torch.bool)
    
    # Simple but fast approach: randomly select spans
    if num_noise_spans > 0:
        # Generate random span starts
        max_start = max(1, length - mean_span_length)
        span_starts = torch.randint(0, max_start, (num_noise_spans,))
        
        tokens_per_span = num_noise_tokens // num_noise_spans
        remaining_tokens = num_noise_tokens % num_noise_spans
        
        for i, start in enumerate(span_starts):
            span_len = tokens_per_span + (1 if i < remaining_tokens else 0)
            end = min(start + span_len, length)
            noise_mask[start:end] = True
    
    # Create input and target sequences
    sentinel_ids = list(range(sentinel_start_id, sentinel_start_id - 100, -1))
    
    # Build input sequence (non-noise tokens + sentinels)
    input_ids = []
    target_ids = []
    sentinel_idx = 0
    
    i = 0
    while i < length:
        if noise_mask[i]:
            # Start of noisy span - add sentinel to input
            if sentinel_idx < len(sentinel_ids):
                input_ids.append(sentinel_ids[sentinel_idx])
                target_ids.append(sentinel_ids[sentinel_idx])
                sentinel_idx += 1
            
            # Add masked tokens to target
            while i < length and noise_mask[i]:
                target_ids.append(tokens[i])
                i += 1
        else:
            # Non-noise token - add to input
            input_ids.append(tokens[i])
            i += 1
    
    # Add EOS token
    eos_id = tokenizer.eos_token_id
    if eos_id is not None:
        input_ids.append(eos_id)
        target_ids.append(eos_id)
    
    return input_ids, target_ids