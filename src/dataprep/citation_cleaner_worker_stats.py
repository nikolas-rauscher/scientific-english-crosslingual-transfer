"""
Citation Cleaner Worker Stats Aggregation Module

Handles multi-worker statistics collection and aggregation for the
MultiCitationCleaner, keeping the main cleaner focused on core functionality.

Usage:
```python
from src.dataprep.citation_cleaner_worker_stats import CitationCleanerWorkerStats

worker_stats = CitationCleanerWorkerStats(
    citation_patterns=citation_patterns,
    max_false_positive_samples=100,
    max_top_citation_docs=50
)

# In run() method:
worker_stats.save_worker_stats(rank, citation_stats, cleaning_stats)
if rank == 0:
    aggregated_stats = worker_stats.aggregate_all_worker_stats(world_size)
```
"""

import logging
import json
import tempfile
import os
import glob
import time
from typing import Dict, Any, Tuple
from collections import defaultdict

log = logging.getLogger(__name__)


class CitationCleanerWorkerStats:
    """Handles worker statistics coordination for multi-task citation cleaning"""
    
    def __init__(
        self,
        citation_patterns: Dict[str, str],
        max_false_positive_samples: int = 100,
        max_top_citation_docs: int = 50,
        enable_smart_validation: bool = True
    ):
        self.citation_patterns = citation_patterns
        self.max_false_positive_samples = max_false_positive_samples
        self.max_top_citation_docs = max_top_citation_docs
        self.enable_smart_validation = enable_smart_validation
    
    def save_worker_stats(self, rank: int, citation_stats: Dict, cleaning_stats: Dict):
        """Speichert Worker-spezifische Stats für spätere Aggregation"""
        try:
            worker_stats = {
                "rank": rank,
                "citation_stats": citation_stats,
                "cleaning_stats": cleaning_stats,
                "enable_smart_validation": self.enable_smart_validation
            }
            
            # Verwende temporäres Verzeichnis
            temp_dir = tempfile.gettempdir()
            stats_dir = os.path.join(temp_dir, "citation_cleaner_stats")
            os.makedirs(stats_dir, exist_ok=True)
            
            stats_file = os.path.join(stats_dir, f"worker_{rank:05d}.json")
            with open(stats_file, 'w') as f:
                json.dump(worker_stats, f, indent=2, default=str)
                
            log.info(f"📊 Worker {rank} stats saved to {stats_file}")
        except Exception as e:
            log.warning(f"Failed to save worker {rank} stats: {e}")
    
    def aggregate_all_worker_stats(self, world_size: int) -> Tuple[Dict, Dict]:
        """Aggregiert Stats von allen Workern für W&B Logging"""
        try:
            # Sammle alle Worker Stats
            temp_dir = tempfile.gettempdir()
            stats_dir = os.path.join(temp_dir, "citation_cleaner_stats")
            
            pattern = os.path.join(stats_dir, "worker_*.json")
            worker_files = glob.glob(pattern)
            
            log.info(f"📊 Aggregating stats from {len(worker_files)} workers (expected: {world_size})")
            
            # Initialize aggregated structures
            aggregated_citation_stats = self._init_aggregated_citation_stats()
            aggregated_cleaning_stats = self._init_aggregated_cleaning_stats()
            
            # Aggregate from all workers
            for worker_file in worker_files:
                try:
                    with open(worker_file, 'r') as f:
                        worker_data = json.load(f)
                    
                    self._merge_worker_data(
                        worker_data, 
                        aggregated_citation_stats, 
                        aggregated_cleaning_stats
                    )
                        
                except Exception as e:
                    log.warning(f"Failed to load worker stats from {worker_file}: {e}")
            
            # Limit and sort collected lists
            self._finalize_aggregated_stats(aggregated_citation_stats, aggregated_cleaning_stats)
            
            # Cleanup temp files
            self._cleanup_temp_files(worker_files, stats_dir)
            
            log.info(f"📊 Successfully aggregated stats from {len(worker_files)} workers: "
                    f"{aggregated_cleaning_stats['docs_processed']} total docs processed")
            
            return aggregated_citation_stats, aggregated_cleaning_stats
            
        except Exception as e:
            log.warning(f"Failed to aggregate worker stats: {e}")
            # Fallback: return empty stats
            return self._init_aggregated_citation_stats(), self._init_aggregated_cleaning_stats()
    
    def _init_aggregated_citation_stats(self) -> Dict:
        """Initialize aggregated citation stats structure"""
        aggregated_citation_stats = {}
        for citation_type in self.citation_patterns.keys():
            aggregated_citation_stats[citation_type] = {
                "docs_with_citations": 0,
                "total_citations_removed": 0,
                "total_citations_found": 0,
                "total_citations_rejected": 0,
                "total_length_reduction": 0,
                "citation_distribution": defaultdict(int),
                "top_citation_docs": [],
                "false_positive_samples": []
            }
        return aggregated_citation_stats
    
    def _init_aggregated_cleaning_stats(self) -> Dict:
        """Initialize aggregated cleaning stats structure"""
        return {
            "docs_processed": 0,
            "docs_with_any_citations": 0,
            "total_citations_all_types": 0,
            "total_citations_rejected": 0,
            "total_length_reduction": 0,
            "total_word_reduction": 0,
            "smart_validation_enabled": self.enable_smart_validation,
            "docs_with_figure_lines_removed": 0,
            "total_figure_lines_removed": 0,
            "total_figure_line_length_reduction": 0,
            "figure_line_removal_samples": [],
            "top_figure_line_removal_docs": [],
            "top_combined_reduction_docs": [],
            "docs_with_citation_limits_exceeded": 0,
            "citation_limit_exceeded_samples": []
        }
    
    def _merge_worker_data(self, worker_data: Dict, aggregated_citation_stats: Dict, aggregated_cleaning_stats: Dict):
        """Merge single worker data into aggregated stats"""
        worker_citation_stats = worker_data["citation_stats"]
        worker_cleaning_stats = worker_data["cleaning_stats"]
        
        # Aggregate cleaning stats
        for key in ["docs_processed", "docs_with_any_citations", "total_citations_all_types",
                   "total_citations_rejected", "total_length_reduction", "total_word_reduction",
                   "docs_with_figure_lines_removed", "total_figure_lines_removed", 
                   "total_figure_line_length_reduction", "docs_with_citation_limits_exceeded"]:
            if key in worker_cleaning_stats:
                aggregated_cleaning_stats[key] += worker_cleaning_stats[key]
        
        # Aggregate citation stats
        for citation_type, stats in worker_citation_stats.items():
            if citation_type in aggregated_citation_stats:
                agg_stats = aggregated_citation_stats[citation_type]
                
                # Sum up numeric fields
                for key in ["docs_with_citations", "total_citations_removed", 
                           "total_citations_found", "total_citations_rejected", "total_length_reduction"]:
                    if key in stats:
                        agg_stats[key] += stats[key]
                
                # Merge distribution
                if "citation_distribution" in stats:
                    for count, freq in stats["citation_distribution"].items():
                        agg_stats["citation_distribution"][int(count)] += freq
                
                # Collect samples (limited)
                if "false_positive_samples" in stats:
                    agg_stats["false_positive_samples"].extend(
                        stats["false_positive_samples"][:10]  # Max 10 per worker
                    )
                
                # Collect top docs (limited)
                if "top_citation_docs" in stats:
                    agg_stats["top_citation_docs"].extend(stats["top_citation_docs"])
        
        # Collect figure removal samples and top docs
        if "figure_line_removal_samples" in worker_cleaning_stats:
            aggregated_cleaning_stats["figure_line_removal_samples"].extend(
                worker_cleaning_stats["figure_line_removal_samples"][:10]
            )
        
        if "top_figure_line_removal_docs" in worker_cleaning_stats:
            aggregated_cleaning_stats["top_figure_line_removal_docs"].extend(
                worker_cleaning_stats["top_figure_line_removal_docs"]
            )
        
        if "top_combined_reduction_docs" in worker_cleaning_stats:
            aggregated_cleaning_stats["top_combined_reduction_docs"].extend(
                worker_cleaning_stats["top_combined_reduction_docs"]
            )
        
        # Collect citation limit exceeded samples
        if "citation_limit_exceeded_samples" in worker_cleaning_stats:
            aggregated_cleaning_stats["citation_limit_exceeded_samples"].extend(
                worker_cleaning_stats["citation_limit_exceeded_samples"][:10]  # Max 10 per worker
            )
    
    def _finalize_aggregated_stats(self, aggregated_citation_stats: Dict, aggregated_cleaning_stats: Dict):
        """Limit and sort collected lists in aggregated stats"""
        for citation_type in aggregated_citation_stats:
            # Limit false positive samples
            aggregated_citation_stats[citation_type]["false_positive_samples"] = \
                aggregated_citation_stats[citation_type]["false_positive_samples"][:self.max_false_positive_samples]
            
            # Sort and limit top docs
            top_docs = aggregated_citation_stats[citation_type]["top_citation_docs"]
            if top_docs:
                aggregated_citation_stats[citation_type]["top_citation_docs"] = \
                    sorted(top_docs, reverse=True)[:self.max_top_citation_docs]
        
        # Sort and limit figure removal lists
        aggregated_cleaning_stats["figure_line_removal_samples"] = \
            aggregated_cleaning_stats["figure_line_removal_samples"][:self.max_false_positive_samples]
        
        if aggregated_cleaning_stats["top_figure_line_removal_docs"]:
            aggregated_cleaning_stats["top_figure_line_removal_docs"] = \
                sorted(aggregated_cleaning_stats["top_figure_line_removal_docs"], reverse=True)[:self.max_top_citation_docs]
        
        if aggregated_cleaning_stats["top_combined_reduction_docs"]:
            aggregated_cleaning_stats["top_combined_reduction_docs"] = \
                sorted(aggregated_cleaning_stats["top_combined_reduction_docs"], reverse=True)[:self.max_top_citation_docs]
        
        # Limit citation limit exceeded samples
        aggregated_cleaning_stats["citation_limit_exceeded_samples"] = \
            aggregated_cleaning_stats["citation_limit_exceeded_samples"][:self.max_false_positive_samples]
    
    def _cleanup_temp_files(self, worker_files: list, stats_dir: str):
        """Clean up temporary worker stats files"""
        try:
            for worker_file in worker_files:
                os.remove(worker_file)
            os.rmdir(stats_dir)
        except:
            pass  # Ignore cleanup errors
    
    def wait_for_workers(self, world_size: int, max_wait_time: int = 10):
        """Wait for all workers to save their stats"""
        log.info(f"📊 Waiting for {world_size} workers to save stats...")
        time.sleep(2)  # Base wait time
        
        # Additional wait if many workers
        if world_size > 10:
            extra_wait = min(5, world_size // 20)
            time.sleep(extra_wait)
            log.info(f"📊 Extended wait for {world_size} workers: +{extra_wait}s") 