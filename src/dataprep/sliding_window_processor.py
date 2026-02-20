import logging
from typing import Dict, List, Tuple
from datatrove.data import Document
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.writers.disk_base import DiskWriter
from datatrove.utils.logging import logger

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Suppress repeated tokenizer length warnings during preprocessing.
# We intentionally tokenize full documents first and then create 512-token windows.
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)


class TextOnlySlidingWindowProcessor(PipelineStep):
    """Creates sliding windows with configurable output format for T5 training."""
    
    type = "sliding_window"
    name = "sliding-window"
    
    def __init__(
        self,
        tokenizer_name_or_path: str = "t5-base",
        target_tokens: int = 512,
        overlap_ratio: float = 0.5,  # 50% overlap
        output_format: str = "text",  # "text" or "tokens"
        log_to_wandb: bool = False,
        wandb_project: str = "BA-DataTrove",
        wandb_group: str = "sliding-windows",
        # Normalization options
        normalize_whitespace: bool = True,
        # Optional per-window filters (threshold-based); pass None/empty to disable
        filters: dict | None = None,
        # Optional: dynamically adjust stride so the last window is not too short
        balance_last_window: bool = False,
        min_last_tokens: int = 384,
    ):
        super().__init__()
        
        self.target_tokens = target_tokens
        self.overlap_ratio = overlap_ratio
        self.overlap_tokens = int(target_tokens * overlap_ratio)
        self.stride_tokens = target_tokens - self.overlap_tokens
        
        if output_format not in ["text", "tokens"]:
            raise ValueError(f"output_format must be 'text' or 'tokens', got: {output_format}")
        self.output_format = output_format
        
        self.log_to_wandb = log_to_wandb and WANDB_AVAILABLE
        
        # Normalization
        self.normalize_whitespace = normalize_whitespace
        
        # Filters
        self.filters_cfg = filters or {}
        self.filters_enabled = bool(self.filters_cfg.get("enable", False))
        # Default thresholds (can be overridden via cfg)
        self.f_min_actual_tokens = self.filters_cfg.get("min_actual_tokens", None)
        self.f_min_char_length = self.filters_cfg.get("min_char_length", None)
        self.f_max_punct_ratio = self.filters_cfg.get("max_punctuation_ratio", None)
        self.f_max_non_alpha_digit_ratio = self.filters_cfg.get("max_non_alpha_digit_ratio", None)
        self.f_max_uppercase_ratio = self.filters_cfg.get("max_uppercase_ratio", None)

        # Dynamic last-window balancing
        self.balance_last_window = bool(balance_last_window)
        self.min_last_tokens = int(min_last_tokens)
        
        # Load tokenizer for precise window boundaries
        try:
            from transformers import T5TokenizerFast
            import transformers
            transformers.logging.set_verbosity_error()
            
            self.tokenizer = T5TokenizerFast.from_pretrained(tokenizer_name_or_path)
            logger.info(f"loaded t5 tokenizer for precise {output_format} windows: {tokenizer_name_or_path}")
        except Exception as e:
            logger.error(f"failed to load tokenizer {tokenizer_name_or_path}: {e}")
            raise
        
        # metrics tracking
        self.processed_docs = 0
        self.total_input_tokens = 0
        self.total_output_windows = 0
        self.short_docs = 0
        self.long_docs = 0
        
        # w&b init
        self.wandb_run = None
        if self.log_to_wandb:
            logger.info("w&b will be initialized per worker rank")
    
    def _compute_window_count(self, token_count: int) -> int:
        """Compute number of windows based on token count.

        Uses the standard formula: 1 + ceil((N - L) / S) for N > L, where
        N is the document token count, L is the target window length and
        S is the stride. This avoids creating a spurious empty window when
        N is an exact multiple of L.
        """
        if token_count <= self.target_tokens:
            return 1
        # 1 + ceil((N - L) / S) implemented with integer arithmetic
        remaining = token_count - self.target_tokens
        return 1 + (remaining + self.stride_tokens - 1) // self.stride_tokens
    
    def _create_sliding_windows(self, doc: Document) -> List[Document]:
        """
        Always use real tokenization for precise window boundaries.
        Output format depends on self.output_format.
        """
        text = doc.text.strip()
        if not text:
            return []
        
        # Always tokenize for precise boundaries
        full_tokens = self.tokenizer.encode(text, add_special_tokens=False)
        token_count = len(full_tokens)
        
        # create sliding windows based on token positions
        window_count = self._compute_window_count(token_count)
        windows = []

        # Pre-compute start positions; optionally rebalance to avoid very short last window
        if token_count <= self.target_tokens:
            start_positions = [0]
        else:
            # default fixed-stride starts
            start_positions = [i * self.stride_tokens for i in range(window_count)]
            # Determine last window length with default starts
            last_start = start_positions[-1]
            last_len = min(self.target_tokens, max(token_count - last_start, 0))
            # Rebalance only if enabled and last is too short
            if self.balance_last_window and window_count > 1 and last_len < self.min_last_tokens:
                # Distribute starts evenly from 0 to (token_count - target)
                max_start = max(token_count - self.target_tokens, 0)
                if max_start == 0:
                    start_positions = [0]
                else:
                    # Evenly space starts; ensures last_start == max_start
                    start_positions = [
                        int(round(i * (max_start / (window_count - 1)))) for i in range(window_count)
                    ]

        for window_idx, start_pos in enumerate(start_positions):
            if token_count <= self.target_tokens:
                # Short document: single window
                start_pos = 0
                end_pos = token_count
                window_tokens = full_tokens
            else:
                # Extract window from computed start
                if start_pos >= token_count:
                    # Safety guard; shouldn't happen with correct starts
                    break
                end_pos = min(start_pos + self.target_tokens, token_count)
                window_tokens = full_tokens[start_pos:end_pos]
            
            # Create window document based on output format
            # Decode text for normalization/quality metrics and filtering (used for both formats)
            decoded_text = self.tokenizer.decode(window_tokens, skip_special_tokens=True)
            if self.normalize_whitespace:
                decoded_text = self._normalize_text(decoded_text)

            # Compute metrics for filtering
            # IMPORTANT: Count tokens on the normalized, stored text so
            # metadata.actual_tokens matches the saved `text` exactly.
            actual_tokens = len(self.tokenizer.encode(decoded_text, add_special_tokens=False))
            quality = self._compute_quality_metrics(decoded_text)

            # Apply filters, if enabled
            if self._should_filter_out(decoded_text, actual_tokens, quality):
                continue

            if self.output_format == "tokens":
                # pad to target_tokens for consistent length
                if len(window_tokens) < self.target_tokens:
                    pad_count = self.target_tokens - len(window_tokens)
                    window_tokens.extend([self.tokenizer.pad_token_id or 0] * pad_count)
                window_tokens = window_tokens[:self.target_tokens]
                
                window_doc = self._create_token_window_document(
                    doc, window_idx, window_tokens, start_pos, end_pos, token_count
                )
                # attach quality metrics
                if isinstance(window_doc.metadata, dict):
                    window_doc.metadata["quality_metrics"] = quality
            else:  # output_format == "text"
                window_doc = self._create_text_window_document(
                    doc, window_idx, decoded_text, start_pos, end_pos, token_count, actual_tokens
                )
                if isinstance(window_doc.metadata, dict):
                    window_doc.metadata["quality_metrics"] = quality

            windows.append(window_doc)
        
        return windows

    def _normalize_text(self, text: str) -> str:
        # Basic whitespace normalization suitable for T5
        if not text:
            return text
        # Replace non-breaking spaces
        text = text.replace("\u00A0", " ")
        # Normalize newlines to spaces
        text = text.replace("\r", "\n").replace("\n", " ")
        # Collapse repeated whitespace
        import re
        text = re.sub(r"\s+", " ", text)
        # Trim
        return text.strip()

    def _compute_quality_metrics(self, text: str) -> dict:
        import string
        total = max(len(text), 1)
        whites = sum(ch.isspace() for ch in text)
        digits = sum(ch.isdigit() for ch in text)
        letters = sum(ch.isalpha() for ch in text)
        uppers = sum(ch.isupper() for ch in text)
        punct = sum(ch in string.punctuation for ch in text)
        non_alpha_digit = sum((not ch.isalnum()) for ch in text)
        return {
            "length": total,
            "white_space_ratio": whites / total,
            "digit_ratio": digits / total,
            "uppercase_ratio": (uppers / letters) if letters > 0 else 0.0,
            "punctuation_ratio": punct / total,
            "non_alpha_digit_ratio": non_alpha_digit / total,
        }

    def _should_filter_out(self, text: str, actual_tokens: int, quality: dict) -> bool:
        if not self.filters_enabled:
            return False
        # Token/length gates
        if self.f_min_actual_tokens is not None and actual_tokens < int(self.f_min_actual_tokens):
            return True
        if self.f_min_char_length is not None and quality.get("length", 0) < int(self.f_min_char_length):
            return True
        # Ratio thresholds
        if self.f_max_punct_ratio is not None and quality.get("punctuation_ratio", 0) > float(self.f_max_punct_ratio):
            return True
        if self.f_max_non_alpha_digit_ratio is not None and quality.get("non_alpha_digit_ratio", 0) > float(self.f_max_non_alpha_digit_ratio):
            return True
        if self.f_max_uppercase_ratio is not None and quality.get("uppercase_ratio", 0) > float(self.f_max_uppercase_ratio):
            return True
        return False
    
    def _create_text_window_document(
        self, 
        original_doc: Document, 
        window_idx: int, 
        window_text: str,
        start_pos: int,
        end_pos: int,
        original_token_count: int,
        actual_tokens: int
    ) -> Document:
        
        # unique window ID
        window_id = f"{original_doc.id}_window_{window_idx:04d}"
        
        window_metadata = {
            "original_doc_id": original_doc.id,
            "window_idx": window_idx,
            
            "start_pos": start_pos,
            "end_pos": end_pos,
            "window_length_chars": len(window_text),
            "actual_tokens": actual_tokens,
            
            "original_token_count": original_token_count,
            "original_window_count": self._compute_window_count(original_token_count),
            
            "window_config": {
                "target_tokens": self.target_tokens,
                "overlap_ratio": self.overlap_ratio,
                "stride_tokens": self.stride_tokens,
                "dynamic_last_window": self.balance_last_window,
                "min_last_tokens": self.min_last_tokens,
                "tokenizer": str(self.tokenizer.name_or_path),
                "mode": "text_precise"
            },
            
            "document_type": "text_sliding_window",
            "preprocessing_version": "v6_precise_text_windows"
        }
        
        # preserve original metadata
        if hasattr(original_doc, 'metadata') and original_doc.metadata:
            window_metadata["original_metadata"] = original_doc.metadata
        
        # Create window document
        window_doc = Document(
            text=window_text,
            id=window_id,
            metadata=window_metadata
        )
        
        return window_doc
    
    def _tokenize_and_create_windows(self, doc: Document) -> List[Document]:
        """
        Main method that dispatches to sliding window creation.
        """
        return self._create_sliding_windows(doc)
    
    def _create_token_window_document(
        self, 
        original_doc: Document, 
        window_idx: int, 
        tokens: List[int],
        start_pos: int,
        end_pos: int,
        original_token_count: int
    ) -> Document:
        
        # unique window ID
        window_id = f"{original_doc.id}_window_{window_idx:04d}"
        
        # tokens as space-separated string
        token_text = " ".join(map(str, tokens))
        
        window_metadata = {
            "original_doc_id": original_doc.id,
            "window_idx": window_idx,
            
            "start_pos": start_pos,
            "end_pos": end_pos,
            "window_length": len(tokens),
            
            "original_token_count": original_token_count,
            "original_window_count": self._compute_window_count(original_token_count),
            
            "window_config": {
                "target_tokens": self.target_tokens,
                "overlap_ratio": self.overlap_ratio,
                "stride_tokens": self.stride_tokens,
                "dynamic_last_window": self.balance_last_window,
                "min_last_tokens": self.min_last_tokens,
                "tokenizer": str(self.tokenizer.name_or_path),
                "mode": "token_based"
            },
            
            "document_type": "token_sliding_window",
            "preprocessing_version": "v5_configurable_windows"
        }
        
        # preserve original metadata
        if hasattr(original_doc, 'metadata') and original_doc.metadata:
            window_metadata["original_metadata"] = original_doc.metadata
        
        # Create window document
        window_doc = Document(
            text=token_text,
            id=window_id,
            metadata=window_metadata
        )
        
        return window_doc
    
    def _log_document_metrics(self, doc: Document, window_count: int, token_count: int):
        
        self.processed_docs += 1
        self.total_input_tokens += token_count  # Always track tokens now
        self.total_output_windows += window_count
        
        if window_count == 1:
            self.short_docs += 1
        else:
            self.long_docs += 1
        
        # log to w&b periodically
        if self.log_to_wandb and self.processed_docs % 1000 == 0 and hasattr(self, 'wandb_run') and self.wandb_run:
            try:
                avg_windows_per_doc = self.total_output_windows / self.processed_docs
                avg_tokens_per_doc = self.total_input_tokens / self.processed_docs
                
                wandb.log({
                    "processed_documents": self.processed_docs,
                    "total_sliding_windows": self.total_output_windows,
                    "short_documents": self.short_docs,
                    "long_documents": self.long_docs,
                    "avg_windows_per_doc": avg_windows_per_doc,
                    "avg_tokens_per_doc": avg_tokens_per_doc,
                    "compression_ratio": self.total_output_windows / self.processed_docs,
                    "long_doc_ratio": self.long_docs / self.processed_docs,
                    "output_format": self.output_format
                })
            except Exception as e:
                logger.warning(f"w&b logging failed: {e}")
                self.log_to_wandb = False
    
    def run(self, data, rank: int = 0, world_size: int = 1):
        
        if self.log_to_wandb and rank == 0 and self.wandb_run is None:
            try:
                self.wandb_run = wandb.init(
                    project="BA-DataTrove",
                    group="sliding-windows", 
                    job_type=f"sliding-window-creation-rank-{rank}",
                    tags=["preprocessing", "sliding-windows", "t5"],
                    config={
                        "target_tokens": self.target_tokens,
                        "overlap_ratio": self.overlap_ratio,
                        "stride_tokens": self.stride_tokens,
                        "tokenizer": str(self.tokenizer.name_or_path),
                        "output_format": self.output_format,
                        "rank": rank,
                        "world_size": world_size
                    },
                    reinit=True
                )
                logger.info(f"w&b initialized for rank {rank}: {self.wandb_run.url}")
            except Exception as e:
                logger.warning(f"w&b initialization failed for rank {rank}: {e}")
                self.log_to_wandb = False
        elif rank != 0:
            self.log_to_wandb = False
            
        logger.info(f"starting sliding window creation (rank {rank}/{world_size})")
        logger.info(f"config: target_tokens={self.target_tokens}, overlap={self.overlap_ratio:.1%}")
        logger.info(f"mode: {'token-based (precise)' if self.output_format == 'tokens' else 'text-based (fast)'}")
        logger.info(f"tokenizer: {self.tokenizer.name_or_path}")
        
        for doc in data:
            self.stat_update("input_documents")
            
            with self.track_time():
                try:
                    # create sliding windows for this document
                    window_docs = self._tokenize_and_create_windows(doc)
                    
                    # update metrics - we always tokenize now
                    token_count = len(self.tokenizer.encode(doc.text, add_special_tokens=False))
                    window_count = len(window_docs)
                    
                    self._log_document_metrics(doc, window_count, token_count)
                    
                    # update stats
                    self.stat_update("output_windows", window_count)
                    self.stat_update("tokens_processed", token_count)
                    
                    if window_count == 1:
                        self.stat_update("short_documents")
                    else:
                        self.stat_update("long_documents")
                    
                    for window_doc in window_docs:
                        yield window_doc
                    
                except Exception as e:
                    logger.warning(f"error processing document {doc.id}: {e}")
                    self.stat_update("processing_errors")
                    continue
        
        # final summary
        if self.processed_docs > 0:
            avg_windows = self.total_output_windows / self.processed_docs
            avg_tokens = self.total_input_tokens / self.processed_docs
            compression_ratio = self.total_output_windows / self.processed_docs
            
            logger.info(f"sliding window creation completed!")
            logger.info(f"output format: {self.output_format}")
            logger.info(f"processed {self.processed_docs} input documents")
            logger.info(f"generated {self.total_output_windows} sliding windows")
            logger.info(f"compression ratio: {compression_ratio:.1f}x (windows per document)")
            logger.info(f"average tokens per document: {avg_tokens:.0f}")
            logger.info(f"average windows per document: {avg_windows:.1f}")
            logger.info(f"short docs: {self.short_docs} ({self.short_docs/self.processed_docs:.1%})")
            logger.info(f"long docs: {self.long_docs} ({self.long_docs/self.processed_docs:.1%})")
            
            # final w&b log
            if self.log_to_wandb and hasattr(self, 'wandb_run') and self.wandb_run and rank == 0:
                try:
                    wandb.log({
                        "final_input_documents": self.processed_docs,
                        "final_sliding_windows": self.total_output_windows,
                        "final_compression_ratio": compression_ratio,
                        "final_avg_windows_per_doc": avg_windows,
                        "final_avg_tokens_per_doc": avg_tokens,
                        "final_short_doc_ratio": self.short_docs / self.processed_docs,
                        "final_output_format": self.output_format
                    })
                    wandb.finish()
                    logger.info(f"📊 W&B session finished for rank {rank}")
                except Exception as e:
                    logger.warning(f"Final W&B logging failed: {e}")


# Keep old class name for compatibility
SlidingWindowProcessor = TextOnlySlidingWindowProcessor
