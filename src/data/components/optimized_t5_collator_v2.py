"""
DataCollatorForT5MLM with backwards compatibility for running experiments.
Supports both legacy (1.5x heuristic) and optimized (exact T5 formula) modes.
"""

from typing import Dict, List
import numpy as np
from transformers import (
    BatchEncoding,
    PreTrainedTokenizerBase,
)
from dataclasses import dataclass
import torch


def shift_tokens_right(input_ids: np.array, pad_token_id: int, decoder_start_token_id: int) -> np.ndarray:
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = np.zeros_like(input_ids)
    shifted_input_ids[:, 1:] = input_ids[:, :-1]
    shifted_input_ids[:, 0] = decoder_start_token_id

    shifted_input_ids = np.where(shifted_input_ids == -100, pad_token_id, shifted_input_ids)
    return shifted_input_ids

def compute_t5_input_and_target_lengths(inputs_length, noise_density, mean_noise_span_length):
    """This function is copy of `random_spans_helper <https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/data/preprocessors.py#L2466>`__ .
    Training parameters to avoid padding with random_spans_noise_mask.
    When training a model with random_spans_noise_mask, we would like to set the other
    training hyperparmeters in a way that avoids padding.
    This function helps us compute these hyperparameters.
    We assume that each noise span in the input is replaced by extra_tokens_per_span_inputs sentinel tokens,
    and each non-noise span in the targets is replaced by extra_tokens_per_span_targets sentinel tokens.
    This function tells us the required number of tokens in the raw example (for split_tokens())
    as well as the length of the encoded targets. Note that this function assumes
    the inputs and targets will have EOS appended and includes that in the reported length.
    Args:
        inputs_length: an integer - desired length of the tokenized inputs sequence
        noise_density: a float
        mean_noise_span_length: a float
    Returns:
        tokens_length: length of original text in tokens
        targets_length: an integer - length in tokens of encoded targets sequence
    """

    def _tokens_length_to_inputs_length_targets_length(tokens_length):
        num_noise_tokens = int(round(tokens_length * noise_density))
        num_nonnoise_tokens = tokens_length - num_noise_tokens
        num_noise_spans = int(round(num_noise_tokens / mean_noise_span_length))
        # inputs contain all nonnoise tokens, sentinels for all noise spans
        # and one EOS token.
        _input_length = num_nonnoise_tokens + num_noise_spans + 1
        _output_length = num_noise_tokens + num_noise_spans + 1
        return _input_length, _output_length

    tokens_length = inputs_length

    while _tokens_length_to_inputs_length_targets_length(tokens_length + 1)[0] <= inputs_length:
        tokens_length += 1

    inputs_length, targets_length = _tokens_length_to_inputs_length_targets_length(tokens_length)

    # minor hack to get the targets length to be equal to inputs length
    # which is more likely to have been set to a nice round number.
    if noise_density == 0.5 and targets_length > inputs_length:
        tokens_length -= 1
        targets_length -= 1
    return tokens_length, targets_length


@dataclass
class DataCollatorForT5MLM:
    """
    Data collator used for T5 span-masked language modeling.
    Supports both legacy (1.5x heuristic) and optimized (exact T5 formula) modes.
    
    Args:
        tokenizer: The tokenizer used for encoding the data.
        noise_density: The probability with which to (randomly) mask tokens in the input.
        mean_noise_span_length: The average span length of the masked tokens.
        input_length: The expected input length after masking.
        target_length: The expected target length after masking.
        pad_token_id: The pad token id of the model
        decoder_start_token_id: The decoder start token id of the model
        use_legacy_heuristic: If True, use 1.5x expansion (for backwards compatibility).
                             If False, use exact T5 formula (recommended for new runs).
    """

    tokenizer: PreTrainedTokenizerBase
    noise_density: float
    mean_noise_span_length: float
    input_length: int
    target_length: int
    pad_token_id: int
    decoder_start_token_id: int
    use_legacy_heuristic: bool = True  # Default to legacy for backwards compatibility

    def __post_init__(self):
        """Validate configuration based on mode."""
        if not self.use_legacy_heuristic:
            # In optimized mode, validate that target_length matches T5 formula
            _, expected_target_length = compute_t5_input_and_target_lengths(
                self.input_length, self.noise_density, self.mean_noise_span_length
            )
            if self.target_length != expected_target_length:
                raise ValueError(
                    f"target_length mismatch: provided {self.target_length}, "
                    f"but T5 formula expects {expected_target_length} for "
                    f"input_length={self.input_length}, noise_density={self.noise_density}, "
                    f"mean_noise_span_length={self.mean_noise_span_length}. "
                    f"Set use_legacy_heuristic=True to use 1.5x expansion instead."
                )
        
        # Get decoder start token robustly (None-safe, since 0 is valid)
        if self.decoder_start_token_id is None:
            self.decoder_start_token_id = getattr(self.tokenizer, "decoder_start_token_id", None)
        if self.decoder_start_token_id is None:
            self.decoder_start_token_id = self.pad_token_id

    def __call__(self, examples: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:

        # Handle both pre-tokenized and text inputs
        processed_examples = []
        
        if self.use_legacy_heuristic:
            # Legacy mode: use 1.5x heuristic (for backwards compatibility with running experiments)
            expanded_length = int(self.input_length * 1.5)
        else:
            # Optimized mode: calculate exact expanded length using T5 formula
            tokens_length, _ = compute_t5_input_and_target_lengths(
                self.input_length, self.noise_density, self.mean_noise_span_length
            )
            expanded_length = tokens_length
        
        for example in examples:
            tokens_field = example.get("tokens", None)
            has_tokens = tokens_field is not None and len(tokens_field) > 0
            if has_tokens:
                # Pre-tokenized input - truncate or pad to expanded length
                tokens = list(tokens_field)
                if len(tokens) > expanded_length:
                    tokens = tokens[:expanded_length]
                elif len(tokens) < expanded_length:
                    # Pad with pad tokens
                    tokens = tokens + [self.pad_token_id] * (expanded_length - len(tokens))
                input_ids = np.array(tokens, dtype=np.int32)
            else:
                # Regular text - tokenize
                text = example.get("text", example.get("content", ""))
                tokenized = self.tokenizer(
                    text,
                    max_length=expanded_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="np",
                    add_special_tokens=False,  # T5 doesn't use special tokens for MLM
                )
                input_ids = tokenized.input_ids[0]
            
            processed_examples.append({"input_ids": input_ids})

        # Normalize every sequence explicitly to a fixed length before stacking.
        # Some legacy rows can carry irregular list/array shapes.
        normalized_input_ids: List[np.ndarray] = []
        for item in processed_examples:
            arr = np.asarray(item["input_ids"], dtype=np.int32).reshape(-1)
            if arr.shape[0] > expanded_length:
                arr = arr[:expanded_length]
            elif arr.shape[0] < expanded_length:
                arr = np.pad(
                    arr,
                    (0, expanded_length - arr.shape[0]),
                    constant_values=self.pad_token_id,
                )
            normalized_input_ids.append(arr)

        # convert list to dict and tensorize input
        batch = BatchEncoding({"input_ids": np.stack(normalized_input_ids, axis=0)})

        input_ids = batch["input_ids"]
        batch_size, expandend_input_length = input_ids.shape

        mask_indices = np.asarray([self.random_spans_noise_mask(expandend_input_length) for i in range(batch_size)])
        labels_mask = ~mask_indices

        input_ids_sentinel = self.create_sentinel_ids(mask_indices.astype(np.int8))
        labels_sentinel = self.create_sentinel_ids(labels_mask.astype(np.int8))
        
        batch["input_ids"] = self.filter_input_ids(input_ids, input_ids_sentinel)
        batch["labels"] = self.filter_input_ids(input_ids, labels_sentinel)

        # Verify/adjust lengths
        input_length = batch["input_ids"].shape[-1]
        labels_length = batch["labels"].shape[-1]
        
        # Handle padding/truncation based on mode
        if self.use_legacy_heuristic:
            # Legacy mode: may need more padding/truncation due to 1.5x heuristic
            if input_length > self.input_length:
                batch["input_ids"] = batch["input_ids"][:, :self.input_length]
            elif input_length < self.input_length:
                pad_width = self.input_length - input_length
                batch["input_ids"] = np.pad(batch["input_ids"], ((0, 0), (0, pad_width)), 
                                          constant_values=self.pad_token_id)
            
            if labels_length > self.target_length:
                batch["labels"] = batch["labels"][:, :self.target_length]
            elif labels_length < self.target_length:
                pad_width = self.target_length - labels_length
                batch["labels"] = np.pad(
                    batch["labels"], ((0, 0), (0, pad_width)), constant_values=self.pad_token_id
                )
        else:
            # Optimized mode: should rarely need padding with correct expanded_length
            if input_length < self.input_length:
                pad_width = self.input_length - input_length
                batch["input_ids"] = np.pad(batch["input_ids"], ((0, 0), (0, pad_width)), 
                                          constant_values=self.pad_token_id)
            elif input_length > self.input_length:
                # This should not happen with correct expanded_length
                batch["input_ids"] = batch["input_ids"][:, :self.input_length]
            
            if labels_length < self.target_length:
                pad_width = self.target_length - labels_length
                batch["labels"] = np.pad(
                    batch["labels"], ((0, 0), (0, pad_width)), constant_values=self.pad_token_id
                )
            elif labels_length > self.target_length:
                # This should not happen with correct expanded_length
                batch["labels"] = batch["labels"][:, :self.target_length]

        # IMPORTANT: Mask label padding positions BEFORE shift (ignore_index = -100)
        batch["labels"] = np.where(batch["labels"] == self.pad_token_id, -100, batch["labels"])
        
        # to check that tokens are correctly preprocessed, one can run `self.tokenizer.batch_decode(input_ids)` and `self.tokenizer.batch_decode(labels)` here...
        batch["decoder_input_ids"] = shift_tokens_right(
            batch["labels"], self.pad_token_id, self.decoder_start_token_id
        )

        # Build attention mask after potential padding
        batch["attention_mask"] = (batch["input_ids"] != self.pad_token_id).astype(np.int32)

        # Convert to tensors with correct dtypes for loss computation
        batch["input_ids"] = torch.as_tensor(batch["input_ids"], dtype=torch.long)
        batch["labels"] = torch.as_tensor(batch["labels"], dtype=torch.long)
        batch["decoder_input_ids"] = torch.as_tensor(batch["decoder_input_ids"], dtype=torch.long)
        batch["attention_mask"] = torch.as_tensor(batch["attention_mask"], dtype=torch.long)
        
        # Comprehensive sanity checks for both modes
        assert batch["input_ids"].shape[-1] == self.input_length, f"Input shape mismatch: got {batch['input_ids'].shape[-1]}, expected {self.input_length}"
        assert batch["labels"].shape[-1] == self.target_length, f"Labels shape mismatch: got {batch['labels'].shape[-1]}, expected {self.target_length}"
        assert (batch["labels"] == self.pad_token_id).sum() == 0, "No PAD tokens should remain in labels (all should be -100)"
        
        # Dtype checks for all tensors
        for key in ("input_ids", "labels", "decoder_input_ids", "attention_mask"):
            assert batch[key].dtype == torch.long, f"{key} should be torch.long, got {batch[key].dtype}"
        
        # Mode-specific validation
        if not self.use_legacy_heuristic:
            # Average token utilization across the batch (fraction of non-PAD tokens)
            nonpad_per_sample = batch["attention_mask"].sum(dim=1)          # [B]
            utilization = (nonpad_per_sample.float() / self.input_length).mean().item()
            if utilization < 0.90:  # Threshold for good utilization
                print(f"WARNING: Low token utilization in optimized mode: {utilization:.3f} "
                      f"(avg non-pad tokens ≈ {nonpad_per_sample.float().mean().item():.1f}/{self.input_length})")

        return batch

    def create_sentinel_ids(self, mask_indices):
        """
        Sentinel ids creation given the indices that should be masked.
        The start indices of each mask are replaced by the sentinel ids in increasing
        order. Consecutive mask indices to be deleted are replaced with `-1`.
        """
        start_indices = mask_indices - np.roll(mask_indices, 1, axis=-1) * mask_indices
        start_indices[:, 0] = mask_indices[:, 0]

        sentinel_ids = np.where(start_indices != 0, np.cumsum(start_indices, axis=-1), start_indices)
        # Use base vocab size (not len(tokenizer) with added tokens) to avoid out-of-range
        # ids for models where tokenizer length exceeds embedding rows (e.g., some T5 variants).
        sentinel_vocab_size = int(getattr(self.tokenizer, "vocab_size", len(self.tokenizer)))
        sentinel_ids = np.where(sentinel_ids != 0, (sentinel_vocab_size - sentinel_ids), 0)
        sentinel_ids -= mask_indices - start_indices

        return sentinel_ids

    def filter_input_ids(self, input_ids, sentinel_ids):
        """
        Puts sentinel mask on `input_ids` and fuse consecutive mask tokens into a single mask token by deleting.
        This will reduce the sequence length from `expanded_inputs_length` to `input_length`.
        """
        batch_size = input_ids.shape[0]

        input_ids_full = np.where(sentinel_ids != 0, sentinel_ids, input_ids)
        # input_ids tokens and sentinel tokens are >= 0, tokens < 0 are
        # masked tokens coming after sentinel tokens and should be removed
        input_ids = input_ids_full[input_ids_full >= 0].reshape((batch_size, -1))
        input_ids = np.concatenate(
            [input_ids, np.full((batch_size, 1), self.tokenizer.eos_token_id, dtype=np.int32)], axis=-1
        )
        return input_ids

    def random_spans_noise_mask(self, length):

        """This function is copy of `random_spans_helper <https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/data/preprocessors.py#L2682>`__ .
        Noise mask consisting of random spans of noise tokens.
        The number of noise tokens and the number of noise spans and non-noise spans
        are determined deterministically as follows:
        num_noise_tokens = round(length * noise_density)
        num_nonnoise_spans = num_noise_spans = round(num_noise_tokens / mean_noise_span_length)
        Spans alternate between non-noise and noise, beginning with non-noise.
        Subject to the above restrictions, all masks are equally likely.
        Args:
            length: an int32 scalar (length of the incoming token sequence)
            noise_density: a float - approximate density of output mask
            mean_noise_span_length: a number
        Returns:
            a boolean tensor with shape [length]
        """

        orig_length = length

        num_noise_tokens = int(np.round(length * self.noise_density))
        # avoid degeneracy by ensuring positive numbers of noise and nonnoise tokens.
        num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
        num_noise_spans = int(np.round(num_noise_tokens / self.mean_noise_span_length))

        # avoid degeneracy by ensuring positive number of noise spans
        num_noise_spans = max(num_noise_spans, 1)
        num_nonnoise_tokens = length - num_noise_tokens

        # pick the lengths of the noise spans and the non-noise spans
        def _random_segmentation(num_items, num_segments):
            """Partition a sequence of items randomly into non-empty segments.
            Args:
                num_items: an integer scalar > 0
                num_segments: an integer scalar in [1, num_items]
            Returns:
                a Tensor with shape [num_segments] containing positive integers that add
                up to num_items
            """
            mask_indices = np.arange(num_items - 1) < (num_segments - 1)
            np.random.shuffle(mask_indices)
            first_in_segment = np.pad(mask_indices, [[1, 0]])
            segment_id = np.cumsum(first_in_segment)
            # count length of sub segments assuming that list is sorted
            _, segment_length = np.unique(segment_id, return_counts=True)
            return segment_length

        noise_span_lengths = _random_segmentation(num_noise_tokens, num_noise_spans)
        nonnoise_span_lengths = _random_segmentation(num_nonnoise_tokens, num_noise_spans)

        interleaved_span_lengths = np.reshape(
            np.stack([nonnoise_span_lengths, noise_span_lengths], axis=1), [num_noise_spans * 2]
        )
        span_starts = np.cumsum(interleaved_span_lengths)[:-1]
        span_start_indicator = np.zeros((length,), dtype=np.int8)
        span_start_indicator[span_starts] = True
        span_num = np.cumsum(span_start_indicator)
        is_noise = np.equal(span_num % 2, 1)

        return is_noise[:orig_length]
