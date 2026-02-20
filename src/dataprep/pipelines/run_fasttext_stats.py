"""

# Habe ich am ende nicht benutzt, da die schwelle sinn macht nach der analyse. 

FastText Statistics Aggregation Pipeline

Reads cleaned documents and aggregates FastText language filtering statistics
from metadata. Provides detailed W&B analytics for language filtering effectiveness.

Usage:
    python src/dataprep/pipelines/run_fasttext_stats.py
    python src/dataprep/pipelines/run_fasttext_stats.py fasttext_stats.limit_documents=1000
"""

import hydra
import logging
from omegaconf import DictConfig
from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.writers import ParquetWriter

import os, sys
# Ensure project root is on PYTHONPATH for src package
script_dir = os.path.dirname(__file__)
proj_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))
sys.path.insert(0, proj_root)

from src.dataprep.fasttext_stats_aggregator import FastTextStatsAggregator

# Setup logging
log = logging.getLogger(__name__)

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# W&B availability check
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    log.warning("wandb not available - FastText stats will not be logged")


@hydra.main(version_base="1.3", config_path="../../../configs", config_name="fasttext_stats/datatrove")
def main(cfg: DictConfig) -> None:
    log.info("🌍 Starting FastText Statistics Aggregation Pipeline")
    
    # W&B Session Management
    wandb_session = None
    if WANDB_AVAILABLE:
        try:
            wandb_session = wandb.init(
                project="BA-DataTrove",
                group="fasttext-statistics",
                tags=["fasttext-stats", "language-filtering", "datatrove", "aggregation"],
                job_type="fasttext-statistics",
                notes="FastText language filtering statistics aggregation from cleaned document metadata",
                config={
                    "limit_documents": cfg.fasttext_stats.limit_documents,
                    "tasks": cfg.fasttext_stats.tasks,
                    "workers": cfg.fasttext_stats.workers,
                    "fasttext_threshold": cfg.fasttext_stats.fasttext_threshold
                }
            )
            log.info(f"📊 W&B session initialized - project: BA-DataTrove")
        except Exception as e:
            log.warning(f"Failed to initialize W&B session: {e}")
            wandb_session = None
    
    pipeline = [
        ParquetReader(
            data_folder=cfg.fasttext_stats.paths.src_dir,
            glob_pattern=cfg.fasttext_stats.paths.src_pattern,
            limit=cfg.fasttext_stats.limit_documents
        ),
        
        FastTextStatsAggregator(
            fasttext_threshold=cfg.fasttext_stats.fasttext_threshold,
            wandb_project="BA-DataTrove",
            wandb_group="fasttext-language-stats",
            log_to_wandb=bool(wandb_session),
            max_sample_docs=cfg.fasttext_stats.get("max_sample_docs", 50)
        ),
        
        # Optional: Save documents with FastText metadata for further analysis
        ParquetWriter(cfg.fasttext_stats.paths.dst) if cfg.fasttext_stats.get("save_output", False) else None
    ]
    
    # Filter out None components
    pipeline = [component for component in pipeline if component is not None]
    
    log.info(f"📋 FastText Statistics Pipeline: {len(pipeline)} steps")
    log.info(f"   1. Parquet Reader (cleaned documents)")
    log.info(f"   2. FastText Statistics Aggregator (W&B: language filtering metrics)")
    if cfg.fasttext_stats.get("save_output", False):
        log.info(f"   3. Parquet Output (with FastText metadata)")
    
    log.info(f"🔧 Config: {cfg.fasttext_stats.tasks} tasks, {cfg.fasttext_stats.workers} workers")
    log.info(f"📁 Input: {cfg.fasttext_stats.paths.src_dir}")
    log.info(f"📊 FastText Threshold: {cfg.fasttext_stats.fasttext_threshold}")
    
    if cfg.fasttext_stats.get("save_output", False):
        log.info(f"📁 Output: {cfg.fasttext_stats.paths.dst}")
    
    if wandb_session:
        log.info(f"📊 W&B Session: {wandb_session.name}")
    else:
        log.info(f"📊 W&B: Disabled (statistics will not be logged)")
    
    executor = LocalPipelineExecutor(
        pipeline=pipeline,
        tasks=cfg.fasttext_stats.tasks,
        workers=cfg.fasttext_stats.workers,
    )
    
    try:
        executor.run()
        log.info("✅ FastText statistics aggregation pipeline completed successfully!")
        
        # Save statistics to file
        stats_aggregator = None
        for component in pipeline:
            if hasattr(component, 'fasttext_stats'):
                stats_aggregator = component
                break
        
        if stats_aggregator:
            import json
            import os
            
            # Save to Hydra output directory
            stats_output_path = os.path.join(os.getcwd(), "fasttext_statistics.json")
            
            stats_summary = {
                "total_docs_processed": stats_aggregator.fasttext_stats["docs_processed"],
                "docs_kept": stats_aggregator.fasttext_stats["docs_kept"],
                "docs_removed": stats_aggregator.fasttext_stats["docs_removed"],
                "kept_rate": stats_aggregator.fasttext_stats["docs_kept"] / stats_aggregator.fasttext_stats["docs_processed"] if stats_aggregator.fasttext_stats["docs_processed"] > 0 else 0,
                "removed_rate": stats_aggregator.fasttext_stats["docs_removed"] / stats_aggregator.fasttext_stats["docs_processed"] if stats_aggregator.fasttext_stats["docs_processed"] > 0 else 0,
                "total_length_kept": stats_aggregator.fasttext_stats["total_length_kept"],
                "total_length_removed": stats_aggregator.fasttext_stats["total_length_removed"],
                "avg_fasttext_score": sum(stats_aggregator.fasttext_stats["fasttext_scores"]) / len(stats_aggregator.fasttext_stats["fasttext_scores"]) if stats_aggregator.fasttext_stats["fasttext_scores"] else 0,
                "avg_kept_score": sum(stats_aggregator.fasttext_stats["kept_docs_scores"]) / len(stats_aggregator.fasttext_stats["kept_docs_scores"]) if stats_aggregator.fasttext_stats["kept_docs_scores"] else 0,
                "avg_removed_score": sum(stats_aggregator.fasttext_stats["removed_docs_scores"]) / len(stats_aggregator.fasttext_stats["removed_docs_scores"]) if stats_aggregator.fasttext_stats["removed_docs_scores"] else 0,
                "threshold_used": stats_aggregator.fasttext_threshold,
                "sample_kept_docs": stats_aggregator.fasttext_stats["sample_kept_docs"][:10],
                "sample_removed_docs": stats_aggregator.fasttext_stats["sample_removed_docs"][:10],
                "fasttext_score_distribution": {
                    "min": min(stats_aggregator.fasttext_stats["fasttext_scores"]) if stats_aggregator.fasttext_stats["fasttext_scores"] else 0,
                    "max": max(stats_aggregator.fasttext_stats["fasttext_scores"]) if stats_aggregator.fasttext_stats["fasttext_scores"] else 0,
                    "count": len(stats_aggregator.fasttext_stats["fasttext_scores"])
                }
            }
            
            with open(stats_output_path, 'w') as f:
                json.dump(stats_summary, f, indent=2)
            
            log.info(f"📊 FastText statistics saved to: {stats_output_path}")
            log.info(f"📋 Summary: {stats_summary['kept_rate']:.1%} kept ({stats_summary['docs_kept']}/{stats_summary['total_docs_processed']}), avg score: {stats_summary['avg_fasttext_score']:.3f}")
        
        # W&B session cleanup
        if wandb_session:
            try:
                wandb.finish()
                log.info("📊 W&B session finished")
            except Exception as e:
                log.warning(f"Error finishing W&B session: {e}")
                
    except Exception as e:
        log.error(f"❌ Pipeline failed: {e}")
        # W&B session cleanup on error
        if wandb_session:
            try:
                wandb.finish()
            except:
                pass
        raise


if __name__ == "__main__":
    main()