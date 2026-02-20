import os
import sys
import logging
import shutil
import wandb
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# 1) Projekt-Root und lokale Datatrove-Quelle ins PYTHONPATH
script_dir = os.path.dirname(__file__)
proj_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))
sys.path.insert(0, proj_root)
#sys.path.insert(0, os.path.join(proj_root, "external", "datatrove", "src"))  # Uncommented to add datatrove to PYTHONPATH
# sys.path.insert(0, os.path.join(proj_root, "external", "datatrove", "src")) # Auskommentiert, falls nicht benötigt

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.readers import ParquetReader


# Hydra-Logger verwenden
log = logging.getLogger(__name__)

def get_datatrove_logging_dir(pipeline_type: str, limit_docs: int) -> str:
    """Erstellt DataTrove-Logging-Verzeichnis basierend auf Hydra-Output-Dir."""
    try:
        # Hydra-Output-Verzeichnis verwenden
        hydra_cfg = HydraConfig.get()
        hydra_output_dir = hydra_cfg.runtime.output_dir
        
        # Parameter-String für bessere Identifikation
        if limit_docs == -1:
            param_str = "unlimited"
        else:
            param_str = f"limit{limit_docs}"
        
        # DataTrove-Logs als Subdirectory im Hydra-Output
        datatrove_dir = os.path.join(hydra_output_dir, f"datatrove_{pipeline_type}_{param_str}")
        
        log.info(f"Using Hydra output directory: {hydra_output_dir}")
        log.info(f"DataTrove logs will be saved to: {datatrove_dir}")
        
        return datatrove_dir
        
    except Exception as e:
        # Fallback zu altem System falls Hydra-Integration fehlt
        log.warning(f"Could not get Hydra output dir ({e}), using fallback")
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        fallback_dir = f"logs/{timestamp}_{pipeline_type}_{limit_docs}"
        return fallback_dir

def get_dual_output_dirs(cfg: DictConfig):
    """Erstellt sowohl Hydra-basierte als auch zentrale Output-Verzeichnisse."""
    try:
        hydra_cfg = HydraConfig.get()
        hydra_output_dir = hydra_cfg.runtime.output_dir
        
        # Primary: Stats im Hydra-Output (mit vollständiger Historie)
        primary_stats_dir = os.path.join(hydra_output_dir, "stats")
        
        # Secondary: Beachte cfg.stats.paths.output_folder Override
        if hasattr(cfg.stats.paths, 'output_folder'):
            central_stats_dir = cfg.stats.paths.output_folder
        else:
            central_stats_dir = os.path.join(proj_root, "data", "statistics")
        
        os.makedirs(primary_stats_dir, exist_ok=True)
        os.makedirs(central_stats_dir, exist_ok=True)
        
        log.info(f"📁 Primary stats output: {primary_stats_dir}")
        log.info(f"📁 Central stats output: {central_stats_dir}")
        
        return primary_stats_dir, central_stats_dir
        
    except Exception as e:
        log.warning(f"Could not setup dual output ({e}), using fallback")
        if hasattr(cfg.stats.paths, 'output_folder'):
            fallback_dir = cfg.stats.paths.output_folder
        else:
            fallback_dir = os.path.join(proj_root, "data", "statistics")
        os.makedirs(fallback_dir, exist_ok=True)
        return fallback_dir, fallback_dir

def sync_stats_to_central(primary_dir: str, central_dir: str):
    """Synchronisiert Stats vom primären zum zentralen Verzeichnis."""
    try:
        log.info("🔄 Syncing stats to central directory...")
        
        # Kopiere alle Stats-Ordner
        for item in os.listdir(primary_dir):
            src_path = os.path.join(primary_dir, item)
            dst_path = os.path.join(central_dir, item)
            
            if os.path.isdir(src_path):
                if os.path.exists(dst_path):
                    shutil.rmtree(dst_path)  # Entferne alte Version
                shutil.copytree(src_path, dst_path)
                log.info(f"  ✅ Synced {item}")
        
        log.info(f"📋 Stats available at: {central_dir}")
        
    except Exception as e:
        log.error(f"❌ Failed to sync stats: {e}")

@hydra.main(
    version_base="1.3",
    config_path="../../../configs",
    config_name="stats/compute_stats_datatrove"
)
def main(cfg: DictConfig) -> None:
    log.info("🚀 Starting Standard Stats Pipeline")
    log.info("Configuration:")
    log.info(f"\n{OmegaConf.to_yaml(cfg)}")

    # Limit-Wert 
    limit_val = cfg.stats.limit_documents

    # Automatische Tag-Generierung basierend auf Run-Parametern
    base_tags = list(cfg.stats.logger.get("tags", []))
    
    if limit_val != -1:
        base_tags.extend(["test_run", f"limit-{limit_val}"])
    else:
        base_tags.extend(["full-dataset"])

    # Wandb initialisieren mit automatischen Tags
    logger_config = OmegaConf.to_container(cfg.stats.logger, resolve=True)
    logger_config["tags"] = base_tags
    
    wandb_run = wandb.init(
        **logger_config,
        config=OmegaConf.to_container(cfg, resolve=True)
    )
    log.info(f"🌐 Wandb initialized: {wandb_run.url}")

    # Dual-Output-System einrichten
    primary_stats_dir, central_stats_dir = get_dual_output_dirs(cfg)
    
    # Hydra-integriertes Logging-Verzeichnis
    datatrove_logging_dir = get_datatrove_logging_dir("standard_stats", limit_val)
    
    # Pipeline aufbauen
    pipeline = [
        ParquetReader(
            data_folder=cfg.stats.paths.input_folder,
            glob_pattern=cfg.stats.paths.src_pattern,
            limit=limit_val,
            text_key=cfg.stats.reader.text_key,
            id_key=cfg.stats.reader.id_key,
            default_metadata=OmegaConf.to_container(cfg.stats.reader.default_metadata, resolve=True)
        ),
    ]
    
    # Füge die Stats-Module zur Pipeline hinzu
    modules = cfg.stats.pipeline.stats_modules
    added_modules = []
    
    # DocStats
    if hasattr(modules, 'doc_stats'):
        doc_stats = hydra.utils.instantiate(
            modules.doc_stats,
            output_folder=os.path.join(primary_stats_dir, modules.doc_stats.output_folder)
        )
        pipeline.append(doc_stats)
        added_modules.append("DocStats")
    
    # LineStats
    if hasattr(modules, 'line_stats'):
        line_stats = hydra.utils.instantiate(
            modules.line_stats,
            output_folder=os.path.join(primary_stats_dir, modules.line_stats.output_folder)
        )
        pipeline.append(line_stats)
        added_modules.append("LineStats")
    
    # ParagraphStats
    if hasattr(modules, 'paragraph_stats'):
        paragraph_stats = hydra.utils.instantiate(
            modules.paragraph_stats,
            output_folder=os.path.join(primary_stats_dir, modules.paragraph_stats.output_folder)
        )
        pipeline.append(paragraph_stats)
        added_modules.append("ParagraphStats")
    
    # TokenStats
    if hasattr(modules, 'token_stats'):
        token_stats = hydra.utils.instantiate(
            modules.token_stats,
            output_folder=os.path.join(primary_stats_dir, modules.token_stats.output_folder)
        )
        pipeline.append(token_stats)
        added_modules.append("TokenStats")
    
    # PerplexityStats - OSCAR (NumPy 2.0 kompatibel, braucht nur KenLM)
    if hasattr(modules, 'perplexity_stats_oscar'):
        perplexity_stats_oscar = hydra.utils.instantiate(
            modules.perplexity_stats_oscar,
            output_folder=os.path.join(primary_stats_dir, modules.perplexity_stats_oscar.output_folder)
        )
        pipeline.append(perplexity_stats_oscar)
        added_modules.append("PerplexityStats-OSCAR")
    
    # PerplexityStats - Wikipedia (NumPy 2.0 kompatibel, braucht nur KenLM)
    if hasattr(modules, 'perplexity_stats_wikipedia'):
        perplexity_stats_wikipedia = hydra.utils.instantiate(
            modules.perplexity_stats_wikipedia,
            output_folder=os.path.join(primary_stats_dir, modules.perplexity_stats_wikipedia.output_folder)
        )
        pipeline.append(perplexity_stats_wikipedia)
        added_modules.append("PerplexityStats-Wikipedia")
    
    # Optional: Enriched documents mit Stats speichern
    if hasattr(cfg.stats, 'save_enriched_docs') and cfg.stats.save_enriched_docs:
        from datatrove.pipeline.writers import ParquetWriter
        enriched_output = os.path.join(primary_stats_dir, "enriched_documents_statistics_v1")
        pipeline.append(ParquetWriter(enriched_output))
        log.info(f"💾 Enriched documents will be saved to: {enriched_output}")
        added_modules.append("ParquetWriter(enriched)")
    
    log.info(f"📊 Pipeline modules: {', '.join(added_modules)}")
    log.info(f"🔧 Total pipeline steps: {len(pipeline)}")
    
    # Führe die Pipeline aus
    executor = LocalPipelineExecutor(
        pipeline=pipeline,
        tasks=cfg.stats.tasks,
        workers=cfg.stats.workers,
        logging_dir=datatrove_logging_dir
    )
    
    log.info("🏃 Starting pipeline execution...")
    start_time = datetime.now()
    executor.run()
    execution_time = (datetime.now() - start_time).total_seconds()
    log.info("✅ Pipeline completed successfully!")
    
    # Nur wichtigste Pipeline-Metriken für Vergleiche
    wandb.log({
        "execution_time": execution_time,
        "limit_documents": limit_val,
        "active_modules": added_modules,  # Liste der verwendeten Module
        
        # Pipeline-Parameter
        "tasks": cfg.stats.tasks,
        "workers": cfg.stats.workers,
        "save_enriched_docs": cfg.stats.get("save_enriched_docs", False),
        
        # Input-Parameter
        "input_folder": cfg.stats.paths.input_folder,
        "src_pattern": cfg.stats.paths.src_pattern,
        "text_key": cfg.stats.reader.text_key,
        "id_key": cfg.stats.reader.id_key,
    })
    
    # Synchronisiere Stats zum zentralen Verzeichnis
    if primary_stats_dir != central_stats_dir:
        sync_stats_to_central(primary_stats_dir, central_stats_dir)
    
    log.info(f"📁 Primary stats (with history): {primary_stats_dir}")
    log.info(f"📁 Central stats (latest): {central_stats_dir}")
    log.info(f"📋 Logs saved to: {datatrove_logging_dir}")
    wandb.finish()      

if __name__ == "__main__":
    main()