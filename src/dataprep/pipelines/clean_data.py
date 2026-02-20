import hydra
import logging
from omegaconf import DictConfig, OmegaConf
from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.writers import ParquetWriter
from datatrove.pipeline.filters import URLFilter, SamplerFilter
from datatrove.pipeline.stats import DocStats

import os, sys
# Ensure project root is on PYTHONPATH  for src package
script_dir = os.path.dirname(__file__)
proj_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))
sys.path.insert(0, proj_root)

# Import modular text cleaning components
from src.dataprep import MultiCitationCleaner, SymbolTokenNormalizer, TextNormalizer

# Setup logging
log = logging.getLogger(__name__)

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# W&B Session Management
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    log.warning("wandb not available - text cleaning analytics will not be logged")


@hydra.main(version_base="1.3", config_path="../../../configs", config_name="cleaning/datatrove")
def main(cfg: DictConfig) -> None:
    log.info("🚀 Starting Modular Text Cleaning Pipeline")

    # Gemeinsame W&B Session initialisieren
    wandb_session = None
    if WANDB_AVAILABLE:
        try:
            wandb_session = wandb.init(
                project="BA-DataTrove",
                group="text-cleaning-pipeline",
                tags=["citation-cleaning", "symbol-normalization", "text-normalization", "datatrove", "modular-pipeline"],
                job_type="text-processing",
                notes="Modular text cleaning with citation removal, symbol normalization and text normalization analytics",
                config={
                    "limit_documents": cfg.cleaning.cleaning.limit_documents,
                    "tasks": cfg.cleaning.cleaning.tasks,
                    "workers": cfg.cleaning.cleaning.workers,
                    "debug_mode": cfg.cleaning.cleaning.get("debug_mode", False),
                    "citation_pattern": r'\s(?:[\w\-]+(?:\d+)?\s+;\s+)+(?:[\w\-]+(?:\d+)?)(?:\s*;)?\s+',
                    "normalization_enabled": True
                }
            )
            log.info(f"📊 Shared W&B session initialized - project: BA-DataTrove")
        except Exception as e:
            log.warning(f"Failed to initialize shared W&B session: {e}")
            wandb_session = None

    pipeline = [
        ParquetReader(
            data_folder=cfg.cleaning.paths.src_dir, 
            glob_pattern=cfg.cleaning.paths.src_pattern, 
            limit=cfg.cleaning.cleaning.limit_documents
        ),
        

        MultiCitationCleaner(
            replacement='',
            track_changes=True,
            debug_mode=cfg.cleaning.cleaning.get("debug_mode", False),  # Debug mode from config
            wandb_project="BA-DataTrove",
            wandb_group="multi-citation-cleaning",
            log_to_wandb=bool(wandb_session)  # Nur loggen wenn Session vorhanden
        ),
        
        # 2. Symbol Token Normalizer - Normalizes scientific symbols for T5
        SymbolTokenNormalizer(
            debug_mode=cfg.cleaning.cleaning.get("debug_mode", False),
            log_to_wandb=bool(wandb_session)
        ),
        
        # 3. Text Normalizer - Normalisiert Whitespace nach Symbol-Normalization
        TextNormalizer(
            normalize_spaces=True,
            normalize_newlines=True,
            strip_whitespace=True,
            track_changes=True,
            wandb_project="BA-DataTrove",
            wandb_group="text-normalization",
            log_to_wandb=bool(wandb_session)  # Nur loggen wenn Session vorhanden
        ),
        
        # 4. Stats berechnen (nutzt bereinigte Dokumente)
        DocStats(
            output_folder=cfg.cleaning.paths.dst + "/stats", 
            groups_to_compute=["summary"]
        ),
        
        # 5. Cleaned + normalized documents speichern
        ParquetWriter(cfg.cleaning.paths.dst),
    ]

    log.info(f"📋 Modular Pipeline: {len(pipeline)} steps")
    log.info(f"   1. Multi-Citation Cleaning (W&B: citation metrics)")  
    log.info(f"   2. Symbol Token Normalization (W&B: symbol metrics)")
    log.info(f"   3. Text Normalization (W&B: normalization metrics)")
    log.info(f"   4. Document Statistics")
    log.info(f"   5. Parquet Output")
    log.info(f"🔧 Config: {cfg.cleaning.cleaning.tasks} tasks, {cfg.cleaning.cleaning.workers} workers")
    
    # Debug mode info
    debug_mode = cfg.cleaning.cleaning.get("debug_mode", False)
    if debug_mode:
        log.info(f"🐛 DEBUG MODE ENABLED - Text wird mit Debug-Tags markiert statt entfernt")
    else:
        log.info(f"🧹 Normal cleaning mode - Text wird entfernt")
    
    log.info(f"📁 Input: {cfg.cleaning.paths.src_dir}")
    log.info(f"📁 Output: {cfg.cleaning.paths.dst}")
    if wandb_session:
        log.info(f"📊 W&B Session: {wandb_session.name} (shared across modules)")
    else:
        log.info(f"📊 W&B: Disabled (metrics will not be logged)")

    executor = LocalPipelineExecutor(
        pipeline=pipeline,
        tasks=cfg.cleaning.cleaning.tasks,
        workers=cfg.cleaning.cleaning.workers,
    )
    
    try:
        executor.run()
        log.info("✅ Modular text cleaning pipeline completed successfully!")
        
        # W&B Session beenden
        if wandb_session:
            try:
                wandb.finish()
                log.info("📊 Shared W&B session finished")
            except Exception as e:
                log.warning(f"Error finishing W&B session: {e}")
            
    except Exception as e:
        log.error(f"❌ Pipeline failed: {e}")
        # W&B Session auch bei Fehler beenden
        if wandb_session:
            try:
                wandb.finish()
            except:
                pass
        raise


if __name__ == "__main__":
    main()