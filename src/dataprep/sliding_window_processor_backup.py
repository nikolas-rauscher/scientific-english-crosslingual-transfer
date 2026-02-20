
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


class SlidingWindowProcessor(PipelineStep):
    """Materializes sliding windows as separate documents."""
    
    type = "sliding_window_materializer"
    name = "sliding-window-materializer"
    
    def __init__(
        self,
        tokenizer_name_or_path: str = "t5-base",
        max_length: int = 512,
        overlap_size: int = 256,
        log_to_wandb: bool = False,
        wandb_project: str = "BA-DataTrove",
        wandb_group: str = "sliding-window-materialization"
    ):
        super().__init__()
        
        self.max_length = max_length
        self.overlap_size = overlap_size
        self.stride = max_length - overlap_size  # step size between windows
        self.log_to_wandb = log_to_wandb and WANDB_AVAILABLE
        
        # init tokenizer
        try:
            from transformers import T5TokenizerFast
            import transformers
            transformers.logging.set_verbosity_error()
            
            self.tokenizer = T5TokenizerFast.from_pretrained(tokenizer_name_or_path)
            logger.info(f"loaded t5 sentencepiece tokenizer: {tokenizer_name_or_path}")
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
        """compute number of windows based on token count"""
        if token_count <= self.max_length:
            return 1
        else:
            return max(1, (token_count - self.overlap_size) // self.stride + 1)
    
    def _tokenize_and_create_windows(self, doc: Document) -> List[Document]:
        """
        Tokenize document and create separate Document for each sliding window.
        Returns:
            List of Documents, each containing one window's tokens as text
        """
        # Get text content
        text = doc.text
        full_tokens = self.tokenizer.encode(text, add_special_tokens=False)
        token_count = len(full_tokens)
        
        # create sliding windows
        window_count = self._compute_window_count(token_count)
        windows = []
        
        for window_idx in range(window_count):
            if token_count <= self.max_length:
                # Short document: single window
                window_tokens = full_tokens
                start_pos = 0
                end_pos = token_count
            else:
                # Long document: extract window with sliding overlap
                start_pos = window_idx * self.stride
                end_pos = min(start_pos + self.max_length, token_count)
                window_tokens = full_tokens[start_pos:end_pos]
            
            # pad to max_length
            if len(window_tokens) < self.max_length:
                pad_count = self.max_length - len(window_tokens)
                window_tokens.extend([self.tokenizer.pad_token_id or 0] * pad_count)
            
            # Ensure exactly max_length
            window_tokens = window_tokens[:self.max_length]
            
            # Create window document
            window_doc = self._create_window_document(
                doc, window_idx, window_tokens, start_pos, end_pos, token_count
            )
            windows.append(window_doc)
        
        return windows
    
    def _create_window_document(
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
                "max_length": self.max_length,
                "overlap_size": self.overlap_size,
                "stride": self.stride,
                "tokenizer": str(self.tokenizer.name_or_path)
            },
            
            "document_type": "t5_sliding_window",
            "preprocessing_version": "v2_materialized_windows"
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
        self.total_input_tokens += token_count
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
                    "total_materialized_windows": self.total_output_windows,
                    "short_documents": self.short_docs,
                    "long_documents": self.long_docs,
                    "avg_windows_per_doc": avg_windows_per_doc,
                    "avg_tokens_per_doc": avg_tokens_per_doc,
                    "compression_ratio": self.total_output_windows / self.processed_docs,
                    "long_doc_ratio": self.long_docs / self.processed_docs
                })
            except Exception as e:
                logger.warning(f"w&b logging failed: {e}")
                self.log_to_wandb = False
    
    def run(self, data, rank: int = 0, world_size: int = 1):
        
        if self.log_to_wandb and rank == 0 and self.wandb_run is None:
            try:
                self.wandb_run = wandb.init(
                    project="BA-DataTrove",
                    group="sliding-window-materialization", 
                    job_type=f"window-materialization-rank-{rank}",
                    tags=["preprocessing", "materialization", "sliding-windows", "t5"],
                    config={
                        "max_length": self.max_length,
                        "overlap_size": self.overlap_size,
                        "stride": self.stride,
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
            
        logger.info(f"starting sliding window materialization (rank {rank}/{world_size})")
        logger.info(f"window config: max_length={self.max_length}, overlap={self.overlap_size}, stride={self.stride}")
        logger.info(f"output: materialized windows as separate documents")
        
        for doc in data:
            self.stat_update("input_documents")
            
            with self.track_time():
                try:
                    # create windows for this document
                    window_docs = self._tokenize_and_create_windows(doc)
                    
                    # update metrics
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
            
            logger.info(f"sliding window materialization completed!")
            logger.info(f"processed {self.processed_docs} input documents")
            logger.info(f"generated {self.total_output_windows} output windows")
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
                        "final_output_windows": self.total_output_windows,
                        "final_compression_ratio": compression_ratio,
                        "final_avg_windows_per_doc": avg_windows,
                        "final_avg_tokens_per_doc": avg_tokens,
                        "final_short_doc_ratio": self.short_docs / self.processed_docs
                    })
                    wandb.finish()
                    logger.info(f"📊 W&B session finished for rank {rank}")
                except Exception as e:
                    logger.warning(f"Final W&B logging failed: {e}") 