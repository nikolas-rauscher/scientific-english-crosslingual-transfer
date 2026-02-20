from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import List, Tuple

import pyarrow as pa
import pyarrow.parquet as pq
from torch.utils.data import Dataset

from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class _ParquetTextFile:
    """Random access to parquet file with efficient row-wise access and LRU cache."""

    def __init__(self, path: Path, max_cached_rowgroups: int = 200):
        # memory_map enables OS-level lazy loading
        self.file = pq.ParquetFile(path, memory_map=True)
        self.num_rows = self.file.metadata.num_rows
        # Large cache for better performance - we have 600GB RAM available
        self._cache = OrderedDict()  # LRU cache for row groups
        self._max_cache_size = max_cached_rowgroups

    def __len__(self):
        return self.num_rows

    def __getitem__(self, idx: int) -> dict:
        # find row group and local index
        row_group, row_index = self._locate(idx)
        
        # Check LRU cache first
        table = self._cache.get(row_group)
        if table is None:
            # Read row group as Arrow table (stays in columnar format)
            table = self.file.read_row_group(row_group, columns=["text", "id", "metadata"])
            # LRU update: move to end (most recent)
            self._cache[row_group] = table
            # Remove oldest if cache full
            if len(self._cache) > self._max_cache_size:
                self._cache.popitem(last=False)  # Remove oldest (FIFO)
        else:
            # Move to end (mark as recently used)
            self._cache.move_to_end(row_group)
        
        # Extract scalar values directly from Arrow columns (zero-copy)
        col_text = table.column("text")[row_index].as_py()
        col_id = table.column("id")[row_index].as_py()
        col_meta = table.column("metadata")[row_index].as_py()
        
        return {"text": col_text, "id": col_id, "metadata": col_meta}

    def _locate(self, global_idx: int):
        rows_cum = 0
        for rg_idx in range(self.file.num_row_groups):
            rg_rows = self.file.metadata.row_group(rg_idx).num_rows
            if rows_cum + rg_rows > global_idx:
                return rg_idx, global_idx - rows_cum
            rows_cum += rg_rows
        raise IndexError("Row index out of range")


class T5ParquetDataset(Dataset):
    """Concatenated parquet files dataset."""

    def __init__(self, parquet_files: List[str | Path], max_cached_rowgroups: int = 2):
        super().__init__()
        self.files = [_ParquetTextFile(Path(f), max_cached_rowgroups) for f in parquet_files]
        self.cumulative_lengths = []
        total = 0
        for f in self.files:
            total += len(f)
            self.cumulative_lengths.append(total)

    def __len__(self):
        return self.cumulative_lengths[-1]

    def _find_file(self, idx: int):
        for file_idx, cum_len in enumerate(self.cumulative_lengths):
            if idx < cum_len:
                prev_cum = 0 if file_idx == 0 else self.cumulative_lengths[file_idx - 1]
                return file_idx, idx - prev_cum
        raise IndexError("Index out of range")

    def __getitem__(self, idx: int):
        file_idx, local_idx = self._find_file(idx)
        row_data = self.files[file_idx][local_idx]
        return row_data






class T5MaterializedWindowDataset(Dataset):
    """Text-only sliding windows dataset for efficient batch tokenization with resume support."""
    
    def __init__(self, parquet_files: List[str | Path], limit_documents: int = -1, max_cached_rowgroups: int = 2):
        super().__init__()
        
        self.base_dataset = T5ParquetDataset(parquet_files, max_cached_rowgroups)
        self._verify_text_windows()
        
        # Dataset length for external references (resume calculation)
        self.total_dataset_length = len(self.base_dataset)
        
        # Apply document limit if specified
        self.limit_documents = limit_documents
        if limit_documents > 0 and limit_documents < len(self.base_dataset):
            self.actual_length = limit_documents
            log.info(f"T5MaterializedWindowDataset created with {limit_documents} text windows (limited from {len(self.base_dataset)})")
        else:
            self.actual_length = len(self.base_dataset)
            log.info(f"T5MaterializedWindowDataset created with {len(self.base_dataset)} text windows")
        
    def _verify_text_windows(self):
        if len(self.base_dataset) == 0:
            raise ValueError("No data found in parquet files")
        
        sample_size = min(5, len(self.base_dataset))
        text_window_count = 0
        
        for i in range(sample_size):
            doc_data = self.base_dataset[i]
            metadata = doc_data.get("metadata", {})
            
            # Check for text window markers
            doc_type = metadata.get("document_type")
            version = metadata.get("preprocessing_version")
            
            if doc_type == "text_sliding_window" and version in ["v3_text_only_windows", "v6_precise_text_windows"]:
                text_window_count += 1
            elif doc_type == "token_sliding_window" and version == "v5_configurable_windows":
                # Token-based sliding windows are also acceptable
                text_window_count += 1
                log.info(f"Found token-based sliding window at index {i}")
            elif doc_type == "t5_sliding_window" and version == "v2_materialized_windows":
                # Legacy pre-tokenized windows are also acceptable
                text_window_count += 1
                log.info(f"Found legacy pre-tokenized window at index {i}")
            
            # Verify text exists
            text = doc_data.get("text", "")
            if not text.strip():
                log.warning(f"Document {i} has empty text")
        
        # Check if we found valid windows
        if text_window_count == 0:
            raise ValueError(
                "No sliding windows found! "
                "Expected metadata.document_type='text_sliding_window' or 't5_sliding_window'. "
                "Run sliding window creation first: "
                "python src/dataprep/pipelines/run_sliding_windows.py"
            )
        
        window_ratio = text_window_count / sample_size
        log.info(f"✅ Verified sliding windows: {text_window_count}/{sample_size} ({window_ratio:.1%})")
        
        if window_ratio < 1.0:
            log.warning(f"⚠️ Only {window_ratio:.1%} of documents are sliding windows. "
                       f"Consider filtering or re-running window creation.")
    
    def __len__(self):
        return self.actual_length
    
    def __getitem__(self, idx: int) -> dict:
        """Get text window with resume support - exact sample position tracking."""
        
        if idx >= self.actual_length:
            raise IndexError(f"Index {idx} out of range for dataset of size {self.actual_length}")
        
        # Apply document limit check
        if self.limit_documents > 0 and idx >= self.limit_documents:
            raise IndexError(f"Index {idx} exceeds document limit {self.limit_documents}")
        
        # Direct index access (no offset - handled by sampler)
        doc_data = self.base_dataset[idx]
        text = doc_data.get("text", "")
        metadata = doc_data.get("metadata", {})
        
        # Check if this is legacy pre-tokenized data
        if metadata.get("document_type") == "t5_sliding_window":
            # Legacy: try to parse as tokens first, fallback to text
            try:
                tokens = [int(x) for x in text.split()[:10]]  # Test first 10
                # If successful, it's pre-tokenized - convert back to text
                # This is a fallback for compatibility
                return {
                    "tokens": [int(x) for x in text.split()],
                    "text": None,  # Signal this is pre-tokenized
                    "type": "legacy_t5_materialized_window",
                    "doc_id": doc_data.get("id", f"unknown_{idx}"),
                    "original_doc_id": metadata.get("original_doc_id"),
                    "window_idx": metadata.get("window_idx", 0),
                    "original_metadata": metadata.get("original_metadata", {})
                }
            except (ValueError, AttributeError):
                # Not pre-tokenized, treat as text
                pass
        
        # Modern text-only window (fast path)
        return {
            "text": text.strip(),
            "type": "text_sliding_window", 
            "doc_id": doc_data.get("id", f"unknown_{idx}"),
            "original_doc_id": metadata.get("original_doc_id"),
            "window_idx": metadata.get("window_idx", 0),
            "original_metadata": metadata.get("original_metadata", {})
        }


 
