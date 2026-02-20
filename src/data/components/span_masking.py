from __future__ import annotations



#this is for reference, we use the transformers.DataCollatorForT5MLM


from typing import List, Tuple

import numpy as np
from transformers.utils import logging

logger = logging.get_logger(__name__)


def _random_segmentation(num_items: int, num_segments: int) -> List[int]:
    """Partition *num_items* into *num_segments* random segments."""
    mask = np.zeros(num_items - 1, dtype=np.int32)
    mask[: num_segments - 1] = 1
    np.random.shuffle(mask)
    segment_lengths = np.diff(np.where(np.concatenate(([1], mask, [1])))[0])
    return segment_lengths.tolist()


def _random_spans_noise_mask(
    length: int, corruption_rate: float, mean_span_length: float
) -> np.ndarray:
    """Produce boolean mask indicating which token positions belong to noisy spans."""
    num_noise_tokens = int(round(length * corruption_rate))
    num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)

    num_noise_spans = int(round(num_noise_tokens / mean_span_length))
    num_noise_spans = max(num_noise_spans, 1)

    num_nonnoise_tokens = length - num_noise_tokens
    # pick lengths for the noise and non-noise segments
    noise_span_lengths = _random_segmentation(num_noise_tokens, num_noise_spans)
    nonnoise_span_lengths = _random_segmentation(num_nonnoise_tokens, num_noise_spans)

    interleaved = []
    for nonnoise, noise in zip(nonnoise_span_lengths, noise_span_lengths):
        interleaved.extend([0] * nonnoise)
        interleaved.extend([1] * noise)
    interleaved.extend([0] * nonnoise_span_lengths[-1])

    mask = np.array(interleaved, dtype=np.bool_)[:length]
    assert mask.sum() == num_noise_tokens, "Masking ratio mismatch"
    return mask


def apply_span_corruption(
    tokens: List[int],
    tokenizer,  # Hugging Face tokenizer with <extra_id_*> tokens
    corruption_rate: float = 0.15,
    mean_span_length: int = 3,
    sentinel_start_id: int | None = None,
) -> Tuple[List[int], List[int]]:
    """Generate *(input_ids, label_ids)* for T5 span corruption.

    Parameters
    ----------
    tokens
        The original tokenised sequence (without special tokens, e.g. <pad>).  
    tokenizer
        A *PreTrainedTokenizer* instance that contains the sentinel tokens
        `<extra_id_0>` … `<extra_id_99>`.
    corruption_rate
        Fraction of tokens to mask.
    mean_span_length
        Average length of the masked spans.
    sentinel_start_id
        Token ID to start counting sentinels from.  Defaults to tokenizer's
        `<extra_id_0>`.
    """
    if sentinel_start_id is None:
        sentinel_start_id = tokenizer.convert_tokens_to_ids("<extra_id_0>")

    tokens = np.array(tokens, dtype=np.int64)
    noise_mask = _random_spans_noise_mask(len(tokens), corruption_rate, mean_span_length)

    # The sentinel tokens are counted in descending order (<extra_id_0> for the first span).
    sentinel_ids = list(range(sentinel_start_id, sentinel_start_id - 100, -1))
    sentinel_idx = 0

    def _create_sentinel_ids(mask: np.ndarray) -> np.ndarray:
        nonlocal sentinel_idx
        output = []
        i = 0
        while i < len(mask):
            if mask[i]:
                # start of a noisy span -> insert a sentinel token
                output.append(sentinel_ids[sentinel_idx])
                sentinel_idx += 1
                # skip until span end
                while i < len(mask) and mask[i]:
                    i += 1
                continue
            output.append(tokens[i])
            i += 1
        return np.array(output, dtype=np.int64)

    input_ids = _create_sentinel_ids(~noise_mask)
    target_ids = _create_sentinel_ids(noise_mask)

    # append eos token
    eos_id = tokenizer.eos_token_id
    if eos_id is not None:
        input_ids = np.concatenate([input_ids, [eos_id]])
        target_ids = np.concatenate([target_ids, [eos_id]])

    return input_ids.tolist(), target_ids.tolist() 