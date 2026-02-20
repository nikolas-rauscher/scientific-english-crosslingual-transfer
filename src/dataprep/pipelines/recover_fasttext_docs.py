"""
# Habe ich am ende nicht benutzt, da die schwelle sinn macht nach der analyse. 

FastText Document Recovery Pipeline

Recovers documents that were filtered out due to FastText threshold being too high.
Extracts documents with scores between old_threshold and new_threshold from original data.

Usage:
    python src/dataprep/pipelines/recover_fasttext_docs.py
    python src/dataprep/pipelines/recover_fasttext_docs.py recovery.old_threshold=0.75 recovery.new_threshold=0.70
"""

import hydra
import logging
from omegaconf import DictConfig
from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.writers import ParquetWriter
from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.data import Document

import os, sys
# Ensure project root is on PYTHONPATH for src package
script_dir = os.path.dirname(__file__)
proj_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))
sys.path.insert(0, proj_root)

# Setup logging
log = logging.getLogger(__name__)

# Load environment variables
from dotenv import load_dotenv
load_dotenv()


class FastTextDocumentRecovery(BaseFilter):
    """
    Recovers documents that were filtered out by FastText threshold.
    Only keeps documents with scores between old and new threshold.
    """
    
    name = "🔄 FastText Document Recovery"
    
    def __init__(
        self,
        old_threshold: float = 0.75,
        new_threshold: float = 0.70,
        debug_mode: bool = False
    ):
        """
        Args:
            old_threshold: Previous threshold that filtered out documents
            new_threshold: New lower threshold
            debug_mode: Debug mode for testing
        """
        super().__init__()
        
        self.old_threshold = old_threshold
        self.new_threshold = new_threshold
        self.debug_mode = debug_mode
        
        # Statistics (don't override self.stats - it's used by DataTrove)
        self.recovery_stats = {
            "docs_processed": 0,
            "docs_recovered": 0,
            "docs_skipped_too_high": 0,
            "docs_skipped_too_low": 0,
            "recovered_docs_info": []
        }
        
        log.info(f"FastText Recovery: {new_threshold} <= score < {old_threshold}")
    
    def filter(self, doc: Document) -> bool:
        """Filter to recover documents in the target FastText score range"""
        self.recovery_stats["docs_processed"] += 1
        
        # Extract FastText score
        fasttext_score = doc.metadata.get('fasttext_en', 1.0)
        
        # Check if document is in recovery range
        if self.new_threshold <= fasttext_score < self.old_threshold:
            # This document should be recovered!
            self.recovery_stats["docs_recovered"] += 1
            
            # Add recovery metadata
            doc.metadata["recovered_by_fasttext"] = True
            doc.metadata["recovery_reason"] = f"fasttext_score_{fasttext_score:.3f}_between_{self.new_threshold}_and_{self.old_threshold}"
            doc.metadata["original_fasttext_score"] = fasttext_score
            
            # Track sample documents
            if len(self.recovery_stats["recovered_docs_info"]) < 20:
                self.recovery_stats["recovered_docs_info"].append({
                    "doc_id": str(doc.id),
                    "fasttext_score": fasttext_score,
                    "text_length": len(doc.text),
                    "text_preview": doc.text[:100] + "..." if len(doc.text) > 100 else doc.text
                })
            
            if self.debug_mode:
                log.info(f"RECOVERED: {doc.id} (score: {fasttext_score:.3f})")
            
            # Pipeline stats
            self.stat_update("docs_recovered")
            
            return True
            
        elif fasttext_score >= self.old_threshold:
            # Document was already kept - skip
            self.recovery_stats["docs_skipped_too_high"] += 1
            self.stat_update("docs_skipped_too_high")
            return False
            
        else:
            # Document score too low even for new threshold - skip  
            self.recovery_stats["docs_skipped_too_low"] += 1
            self.stat_update("docs_skipped_too_low")
            return False
    
    def run(self, data, rank: int = 0, world_size: int = 1):
        """Override run method for final statistics"""
        try:
            yield from super().run(data, rank, world_size)
        finally:
            log.info(f"📊 FastText Recovery rank {rank} completed:")
            log.info(f"   Processed: {self.recovery_stats['docs_processed']} docs")
            log.info(f"   Recovered: {self.recovery_stats['docs_recovered']} docs")
            log.info(f"   Skipped (too high): {self.recovery_stats['docs_skipped_too_high']} docs")
            log.info(f"   Skipped (too low): {self.recovery_stats['docs_skipped_too_low']} docs")
            
            if self.recovery_stats["recovered_docs_info"]:
                log.info(f"📝 Sample recovered documents:")
                for doc_info in self.recovery_stats["recovered_docs_info"][:5]:
                    log.info(f"   {doc_info['doc_id']}: score={doc_info['fasttext_score']:.3f}, "
                            f"length={doc_info['text_length']}")


@hydra.main(version_base="1.3", config_path="../../../configs", config_name="recovery/fasttext")
def main(cfg: DictConfig) -> None:
    log.info("🔄 Starting FastText Document Recovery Pipeline")
    
    pipeline = [
        ParquetReader(
            data_folder=cfg.recovery.paths.src_dir,
            glob_pattern=cfg.recovery.paths.src_pattern,
            limit=cfg.recovery.limit_documents
        ),
        
        FastTextDocumentRecovery(
            old_threshold=cfg.recovery.old_threshold,
            new_threshold=cfg.recovery.new_threshold,
            debug_mode=cfg.recovery.get("debug_mode", False)
        ),
        
        ParquetWriter(cfg.recovery.paths.dst)
    ]
    
    log.info(f"📋 Recovery Pipeline: {len(pipeline)} steps")
    log.info(f"   1. Read original documents from {cfg.recovery.paths.src_dir}")
    log.info(f"   2. Filter FastText scores: {cfg.recovery.new_threshold} <= score < {cfg.recovery.old_threshold}")
    log.info(f"   3. Save recovered documents to {cfg.recovery.paths.dst}")
    
    log.info(f"🔧 Config: {cfg.recovery.tasks} tasks, {cfg.recovery.workers} workers")
    log.info(f"📊 Threshold change: {cfg.recovery.old_threshold} → {cfg.recovery.new_threshold}")
    
    if cfg.recovery.limit_documents > 0:
        log.info(f"🧪 Test mode: Limited to {cfg.recovery.limit_documents} documents")
    
    executor = LocalPipelineExecutor(
        pipeline=pipeline,
        tasks=cfg.recovery.tasks,
        workers=cfg.recovery.workers,
    )
    
    try:
        executor.run()
        log.info("✅ FastText document recovery completed successfully!")
        log.info(f"📁 Recovered documents saved to: {cfg.recovery.paths.dst}")
        log.info(f"🔄 Next step: Run citation cleaning on recovered documents")
        
    except Exception as e:
        log.error(f"❌ Recovery pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()