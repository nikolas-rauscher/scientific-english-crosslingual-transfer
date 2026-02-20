"""
# Habe ich am ende nicht benutzt, da die schwelle sinn macht nach der analyse. 

FastText Statistics Aggregator for DataTrove

Aggregates FastText language cleaning statistics from document metadata.
Since SmartLanguageCleaner already sets metadata fields, this module
reads the metadata and provides aggregated statistics for W&B logging.

Usage:
```python
from src.dataprep.fasttext_stats_aggregator import FastTextStatsAggregator

aggregator = FastTextStatsAggregator(
    fasttext_threshold=0.75,
    log_to_wandb=True
)
```
"""

import logging
from typing import Dict, Any, List, Optional
from datatrove.data import Document
from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.pipeline.writers.disk_base import DiskWriter

log = logging.getLogger(__name__)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    log.warning("wandb not available - FastText stats will not be logged")


class FastTextStatsAggregator(BaseFilter):
    """
    Aggregates FastText language cleaning statistics from document metadata.
    Provides W&B logging and detailed analytics for language filtering.
    """
    
    name = "🌍 FastText Language Statistics Aggregator"
    
    def __init__(
        self,
        fasttext_threshold: float = 0.75,
        wandb_project: str = "BA-DataTrove",
        wandb_group: str = "fasttext-language-stats",
        log_to_wandb: bool = True,
        max_sample_docs: int = 50,
        exclusion_writer: DiskWriter = None
    ):
        """
        Args:
            fasttext_threshold: Expected FastText threshold for reference
            wandb_project: W&B project name
            wandb_group: W&B group for FastText stats
            log_to_wandb: Whether to log stats to W&B
            max_sample_docs: Maximum number of sample documents to track
            exclusion_writer: Optional writer for excluded documents
        """
        super().__init__(exclusion_writer)
        
        self.fasttext_threshold = fasttext_threshold
        self.wandb_project = wandb_project
        self.wandb_group = wandb_group
        self.log_to_wandb = log_to_wandb and WANDB_AVAILABLE
        self.max_sample_docs = max_sample_docs
        
        # Statistics tracking
        self.fasttext_stats = {
            "docs_processed": 0,
            "docs_kept": 0,
            "docs_removed": 0,
            "total_length_kept": 0,
            "total_length_removed": 0,
            "fasttext_scores": [],  # For distribution analysis
            "kept_docs_scores": [],
            "removed_docs_scores": [],
            "sample_kept_docs": [],
            "sample_removed_docs": []
        }
        
        # W&B initialization
        self.wandb_initialized = False
        self.wandb_enabled = log_to_wandb and WANDB_AVAILABLE
        self.current_rank = 0
        
        log.info(f"FastText Stats Aggregator initialized with threshold={fasttext_threshold}")
    
    def filter(self, doc: Document) -> bool:
        """Aggregate FastText statistics from document metadata"""
        self.fasttext_stats["docs_processed"] += 1
        
        # Extract FastText metadata - check for original statistics data
        fasttext_score = doc.metadata.get('fasttext_en', doc.metadata.get('fasttext_score_used', 1.0))
        
        # Simulate language filtering based on threshold (for original data)
        document_kept = fasttext_score >= self.fasttext_threshold
        document_removed = fasttext_score < self.fasttext_threshold
        length_reduction = len(doc.text) if document_removed else 0
        
        # Track score distribution
        self.fasttext_stats["fasttext_scores"].append(fasttext_score)
        
        # Aggregate statistics
        if document_kept and not document_removed:
            self.fasttext_stats["docs_kept"] += 1
            self.fasttext_stats["total_length_kept"] += len(doc.text)
            self.fasttext_stats["kept_docs_scores"].append(fasttext_score)
            
            # Sample kept documents
            if len(self.fasttext_stats["sample_kept_docs"]) < self.max_sample_docs:
                self.fasttext_stats["sample_kept_docs"].append({
                    "doc_id": str(doc.id),
                    "fasttext_score": fasttext_score,
                    "text_length": len(doc.text),
                    "reason": doc.metadata.get('language_removal_reason', 'good_score')
                })
        
        elif document_removed:
            self.fasttext_stats["docs_removed"] += 1
            self.fasttext_stats["total_length_removed"] += length_reduction
            self.fasttext_stats["removed_docs_scores"].append(fasttext_score)
            
            # Sample removed documents
            if len(self.fasttext_stats["sample_removed_docs"]) < self.max_sample_docs:
                self.fasttext_stats["sample_removed_docs"].append({
                    "doc_id": str(doc.id),
                    "fasttext_score": fasttext_score,
                    "original_length": doc.metadata.get('language_original_length', 0),
                    "reason": doc.metadata.get('language_removal_reason', 'low_score')
                })
        
        # W&B logging for main process
        if hasattr(self, 'current_rank') and self.current_rank == 0 and self.wandb_initialized:
            self._log_document_metrics(doc, fasttext_score, document_kept, document_removed)
            
            # Aggregated stats every 100 documents
            if self.fasttext_stats["docs_processed"] % 100 == 0:
                self._log_aggregated_stats()
        
        # Pipeline stats
        self.stat_update("docs_processed")
        if document_kept:
            self.stat_update("docs_kept")
        if document_removed:
            self.stat_update("docs_removed")
        
        # Always pass documents through (this is a stats aggregator, not a filter)
        return True
    
    def _log_document_metrics(self, doc: Document, fasttext_score: float, 
                            document_kept: bool, document_removed: bool):
        """Log per-document FastText metrics to W&B"""
        doc_metrics = {
            "fasttext/score": fasttext_score,
            "fasttext/document_kept": 1 if document_kept else 0,
            "fasttext/document_removed": 1 if document_removed else 0,
            "fasttext/text_length": len(doc.text),
            "fasttext/above_threshold": 1 if fasttext_score >= self.fasttext_threshold else 0
        }
        
        wandb.log(doc_metrics)
    
    def _log_aggregated_stats(self):
        """Log aggregated FastText statistics to W&B"""
        if not self.wandb_initialized:
            return
            
        try:
            stats = self.fasttext_stats
            
            # Calculate aggregated metrics
            total_docs = stats["docs_processed"]
            kept_rate = stats["docs_kept"] / total_docs if total_docs > 0 else 0
            removed_rate = stats["docs_removed"] / total_docs if total_docs > 0 else 0
            
            # Score statistics
            avg_score = sum(stats["fasttext_scores"]) / len(stats["fasttext_scores"]) if stats["fasttext_scores"] else 0
            avg_kept_score = sum(stats["kept_docs_scores"]) / len(stats["kept_docs_scores"]) if stats["kept_docs_scores"] else 0
            avg_removed_score = sum(stats["removed_docs_scores"]) / len(stats["removed_docs_scores"]) if stats["removed_docs_scores"] else 0
            
            agg_stats = {
                "fasttext_agg/docs_processed": total_docs,
                "fasttext_agg/docs_kept": stats["docs_kept"],
                "fasttext_agg/docs_removed": stats["docs_removed"],
                "fasttext_agg/kept_rate": kept_rate,
                "fasttext_agg/removed_rate": removed_rate,
                "fasttext_agg/avg_fasttext_score": avg_score,
                "fasttext_agg/avg_kept_score": avg_kept_score,
                "fasttext_agg/avg_removed_score": avg_removed_score,
                "fasttext_agg/total_length_kept": stats["total_length_kept"],
                "fasttext_agg/total_length_removed": stats["total_length_removed"],
                "fasttext_agg/threshold_used": self.fasttext_threshold
            }
            
            wandb.log(agg_stats)
            
            log.info(f"📊 FastText aggregated stats: {stats['docs_processed']} docs, "
                    f"{stats['docs_kept']} kept ({kept_rate:.1%}), "
                    f"{stats['docs_removed']} removed ({removed_rate:.1%}), "
                    f"avg score: {avg_score:.3f}")
            
        except Exception as e:
            log.warning(f"Failed to log aggregated FastText stats to W&B: {e}")
    
    def _init_wandb(self):
        """Initialize W&B session for FastText statistics"""
        try:
            # Check if W&B session already exists
            if wandb.run is not None:
                self.wandb_initialized = True
                log.info("📊 Using existing W&B session for FastText Stats Aggregator")
            else:
                # Initialize new session
                wandb.init(
                    project=self.wandb_project,
                    group=self.wandb_group,
                    tags=["fasttext-stats", "language-filtering", "datatrove", "aggregation"],
                    job_type="fasttext-statistics",
                    notes="FastText language filtering statistics aggregation from metadata",
                    config={
                        "fasttext_threshold": self.fasttext_threshold,
                        "max_sample_docs": self.max_sample_docs
                    }
                )
                self.wandb_initialized = True
                log.info(f"📊 W&B initialized for FastText Stats - project: {self.wandb_project}")
        except Exception as e:
            log.warning(f"Failed to initialize W&B for FastText stats: {e}")
            self.log_to_wandb = False
    
    def _log_final_summary(self):
        """Log final FastText statistics summary"""
        if not self.wandb_initialized:
            return
            
        stats = self.fasttext_stats
        
        # Final summary metrics
        total_docs = stats["docs_processed"]
        final_stats = {
            "fasttext_summary/total_docs_processed": total_docs,
            "fasttext_summary/total_docs_kept": stats["docs_kept"],
            "fasttext_summary/total_docs_removed": stats["docs_removed"],
            "fasttext_summary/final_kept_rate": stats["docs_kept"] / total_docs if total_docs > 0 else 0,
            "fasttext_summary/final_removed_rate": stats["docs_removed"] / total_docs if total_docs > 0 else 0,
            "fasttext_summary/total_length_kept": stats["total_length_kept"],
            "fasttext_summary/total_length_removed": stats["total_length_removed"],
            "fasttext_summary/threshold_used": self.fasttext_threshold
        }
        
        wandb.log(final_stats)
        
        # Score distribution histograms
        if stats["fasttext_scores"]:
            wandb.log({
                "fasttext_summary/score_distribution": wandb.Histogram(stats["fasttext_scores"]),
                "fasttext_summary/kept_scores_distribution": wandb.Histogram(stats["kept_docs_scores"]) if stats["kept_docs_scores"] else None,
                "fasttext_summary/removed_scores_distribution": wandb.Histogram(stats["removed_docs_scores"]) if stats["removed_docs_scores"] else None
            })
        
        # Sample documents tables
        if stats["sample_kept_docs"]:
            kept_table_data = []
            for i, doc_info in enumerate(stats["sample_kept_docs"][:20], 1):
                kept_table_data.append([
                    i,
                    doc_info["doc_id"],
                    f"{doc_info['fasttext_score']:.3f}",
                    doc_info["text_length"],
                    doc_info["reason"]
                ])
            
            kept_table = wandb.Table(
                columns=["Rank", "Document ID", "FastText Score", "Text Length", "Reason"],
                data=kept_table_data
            )
            wandb.log({"fasttext_summary/sample_kept_documents": kept_table})
        
        if stats["sample_removed_docs"]:
            removed_table_data = []
            for i, doc_info in enumerate(stats["sample_removed_docs"][:20], 1):
                removed_table_data.append([
                    i,
                    doc_info["doc_id"],
                    f"{doc_info['fasttext_score']:.3f}",
                    doc_info["original_length"],
                    doc_info["reason"]
                ])
            
            removed_table = wandb.Table(
                columns=["Rank", "Document ID", "FastText Score", "Original Length", "Reason"],
                data=removed_table_data
            )
            wandb.log({"fasttext_summary/sample_removed_documents": removed_table})
        
        log.info(f"📋 FastText Summary: {stats['docs_kept']}/{total_docs} docs kept "
                f"({stats['docs_kept']/total_docs:.1%}), "
                f"{stats['docs_removed']} removed, "
                f"avg score: {sum(stats['fasttext_scores'])/len(stats['fasttext_scores']):.3f}")
    
    def run(self, data, rank: int = 0, world_size: int = 1):
        """Override run method for W&B initialization and worker aggregation"""
        self.current_rank = rank
        
        # Initialize W&B for main process
        if rank == 0 and self.wandb_enabled:
            self._init_wandb()
        
        try:
            yield from super().run(data, rank, world_size)
        finally:
            # Save worker stats for aggregation
            self._save_worker_stats(rank)
            
            # Local worker logging
            log.info(f"✅ FastText stats aggregation rank {rank} completed: "
                    f"{self.fasttext_stats['docs_processed']} docs processed, "
                    f"{self.fasttext_stats['docs_kept']} kept, "
                    f"{self.fasttext_stats['docs_removed']} removed")
            
            # Aggregation and final logging for main process
            if rank == 0:
                # Wait for other workers
                import time
                time.sleep(2)
                
                # Aggregate stats from all workers
                aggregated_stats = self._aggregate_all_worker_stats(world_size)
                
                if self.wandb_initialized:
                    # Use aggregated stats for final summary
                    original_stats = self.fasttext_stats
                    self.fasttext_stats = aggregated_stats
                    self._log_final_summary()
                    self.fasttext_stats = original_stats
                
                # Save aggregated stats to file
                self._save_aggregated_stats(aggregated_stats)
                
                log.info(f"📊 FastText Aggregated Summary (ALL {world_size} workers): "
                        f"{aggregated_stats['docs_processed']} docs, "
                        f"{aggregated_stats['docs_kept']} kept ({aggregated_stats['docs_kept']/aggregated_stats['docs_processed']:.1%}), "
                        f"{aggregated_stats['docs_removed']} removed ({aggregated_stats['docs_removed']/aggregated_stats['docs_processed']:.1%}), "
                        f"avg score: {sum(aggregated_stats['fasttext_scores'])/len(aggregated_stats['fasttext_scores']):.3f}")
    
    def _save_worker_stats(self, rank: int):
        """Save worker-specific stats for aggregation"""
        try:
            import json
            import tempfile
            import os
            
            worker_stats = {
                "rank": rank,
                "fasttext_stats": self.fasttext_stats
            }
            
            # Use temporary directory
            temp_dir = tempfile.gettempdir()
            stats_dir = os.path.join(temp_dir, "fasttext_worker_stats")
            os.makedirs(stats_dir, exist_ok=True)
            
            stats_file = os.path.join(stats_dir, f"worker_{rank:05d}.json")
            with open(stats_file, 'w') as f:
                json.dump(worker_stats, f, indent=2, default=str)
            
            log.info(f"📊 FastText worker {rank} stats saved")
        except Exception as e:
            log.warning(f"Failed to save FastText worker {rank} stats: {e}")
    
    def _aggregate_all_worker_stats(self, world_size: int):
        """Aggregate FastText stats from all workers"""
        try:
            import json
            import tempfile
            import os
            import glob
            
            # Collect all worker stats
            temp_dir = tempfile.gettempdir()
            stats_dir = os.path.join(temp_dir, "fasttext_worker_stats")
            
            pattern = os.path.join(stats_dir, "worker_*.json")
            worker_files = glob.glob(pattern)
            
            log.info(f"📊 Aggregating FastText stats from {len(worker_files)} workers (expected: {world_size})")
            
            # Initialize aggregated structures
            aggregated_stats = {
                "docs_processed": 0,
                "docs_kept": 0,
                "docs_removed": 0,
                "total_length_kept": 0,
                "total_length_removed": 0,
                "fasttext_scores": [],
                "kept_docs_scores": [],
                "removed_docs_scores": [],
                "sample_kept_docs": [],
                "sample_removed_docs": []
            }
            
            # Aggregate from all workers
            for worker_file in worker_files:
                try:
                    with open(worker_file, 'r') as f:
                        worker_data = json.load(f)
                    
                    worker_stats = worker_data["fasttext_stats"]
                    
                    # Aggregate numeric stats
                    for key in ["docs_processed", "docs_kept", "docs_removed", "total_length_kept", "total_length_removed"]:
                        if key in worker_stats:
                            aggregated_stats[key] += worker_stats[key]
                    
                    # Aggregate lists (with limits)
                    aggregated_stats["fasttext_scores"].extend(worker_stats.get("fasttext_scores", []))
                    aggregated_stats["kept_docs_scores"].extend(worker_stats.get("kept_docs_scores", []))
                    aggregated_stats["removed_docs_scores"].extend(worker_stats.get("removed_docs_scores", []))
                    
                    # Limit sample docs
                    aggregated_stats["sample_kept_docs"].extend(worker_stats.get("sample_kept_docs", []))
                    aggregated_stats["sample_removed_docs"].extend(worker_stats.get("sample_removed_docs", []))
                    
                except Exception as e:
                    log.warning(f"Failed to load FastText worker stats from {worker_file}: {e}")
            
            # Limit sample sizes
            aggregated_stats["sample_kept_docs"] = aggregated_stats["sample_kept_docs"][:self.max_sample_docs]
            aggregated_stats["sample_removed_docs"] = aggregated_stats["sample_removed_docs"][:self.max_sample_docs]
            
            # Cleanup temp files
            try:
                for worker_file in worker_files:
                    os.remove(worker_file)
                os.rmdir(stats_dir)
            except:
                pass  # Ignore cleanup errors
            
            log.info(f"📊 Successfully aggregated FastText stats from {len(worker_files)} workers: "
                    f"{aggregated_stats['docs_processed']} total docs processed")
            
            return aggregated_stats
            
        except Exception as e:
            log.warning(f"Failed to aggregate FastText worker stats: {e}")
            # Fallback: return rank 0 stats
            return self.fasttext_stats
    
    def _save_aggregated_stats(self, aggregated_stats):
        """Save aggregated stats to file"""
        try:
            import json
            import os
            
            stats_output_path = os.path.join(os.getcwd(), "fasttext_statistics_aggregated.json")
            
            stats_summary = {
                "total_docs_processed": aggregated_stats["docs_processed"],
                "docs_kept": aggregated_stats["docs_kept"],
                "docs_removed": aggregated_stats["docs_removed"],
                "kept_rate": aggregated_stats["docs_kept"] / aggregated_stats["docs_processed"] if aggregated_stats["docs_processed"] > 0 else 0,
                "removed_rate": aggregated_stats["docs_removed"] / aggregated_stats["docs_processed"] if aggregated_stats["docs_processed"] > 0 else 0,
                "total_length_kept": aggregated_stats["total_length_kept"],
                "total_length_removed": aggregated_stats["total_length_removed"],
                "avg_fasttext_score": sum(aggregated_stats["fasttext_scores"]) / len(aggregated_stats["fasttext_scores"]) if aggregated_stats["fasttext_scores"] else 0,
                "avg_kept_score": sum(aggregated_stats["kept_docs_scores"]) / len(aggregated_stats["kept_docs_scores"]) if aggregated_stats["kept_docs_scores"] else 0,
                "avg_removed_score": sum(aggregated_stats["removed_docs_scores"]) / len(aggregated_stats["removed_docs_scores"]) if aggregated_stats["removed_docs_scores"] else 0,
                "threshold_used": self.fasttext_threshold,
                "sample_kept_docs": aggregated_stats["sample_kept_docs"][:20],
                "sample_removed_docs": aggregated_stats["sample_removed_docs"][:20],
                "fasttext_score_distribution": {
                    "min": min(aggregated_stats["fasttext_scores"]) if aggregated_stats["fasttext_scores"] else 0,
                    "max": max(aggregated_stats["fasttext_scores"]) if aggregated_stats["fasttext_scores"] else 0,
                    "count": len(aggregated_stats["fasttext_scores"])
                }
            }
            
            with open(stats_output_path, 'w') as f:
                json.dump(stats_summary, f, indent=2)
            
            log.info(f"📊 Aggregated FastText statistics saved to: {stats_output_path}")
            
        except Exception as e:
            log.warning(f"Failed to save aggregated FastText statistics: {e}")