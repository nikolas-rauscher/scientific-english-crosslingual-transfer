"""
Citation Cleaner W&B Logging Module

Separate logging module for MultiCitationCleaner to keep the main cleaner
focused on core functionality while providing comprehensive W&B analytics.

Usage:
```python
from src.dataprep.citation_cleaner_logger import CitationCleanerLogger

logger = CitationCleanerLogger(
    wandb_project="BA-DataTrove",
    wandb_group="smart-multi-citation-cleaning"
)
```
"""

import logging
from typing import Dict, Any, List
from collections import defaultdict
import heapq

log = logging.getLogger(__name__)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    log.warning("wandb not available - citation cleaning stats will not be logged")


class CitationCleanerLogger:
    """Separate W&B logging class for citation cleaning analytics"""
    
    def __init__(
        self,
        wandb_project: str = "BA-DataTrove",
        wandb_group: str = "smart-multi-citation-cleaning",
        log_to_wandb: bool = True,
        max_false_positive_samples: int = 100,
        max_top_citation_docs: int = 50,
    ):
        self.wandb_project = wandb_project
        self.wandb_group = wandb_group
        self.log_to_wandb = log_to_wandb and WANDB_AVAILABLE
        self.max_false_positive_samples = max_false_positive_samples
        self.max_top_citation_docs = max_top_citation_docs
        
        # W&B Initialisierung
        self.wandb_initialized = False
        if self.log_to_wandb:
            self._init_wandb()
    
    def _init_wandb(self):
        """Prüft ob bereits eine W&B Session läuft - nutzt shared session"""
        try:
            # Prüfe ob bereits eine wandb session läuft
            if wandb.run is not None:
                self.wandb_initialized = True
                log.info(f"📊 Using existing W&B session for Citation Cleaner Logger")
            else:
                # Fallback: eigene Session (sollte normalerweise nicht passieren)
                wandb.init(
                    project=self.wandb_project,
                    group=self.wandb_group,
                    tags=["smart-multi-citation-cleaning", "false-positive-prevention", "datatrove", "text-preprocessing"],
                    job_type="data-cleaning",
                    notes="Enhanced multi-type citation cleaning with smart validation to prevent false positives",
                )
                self.wandb_initialized = True
                log.info(f"📊 W&B initialized for Citation Cleaner Logger - project: {self.wandb_project}")
        except Exception as e:
            log.warning(f"Failed to initialize W&B for citation cleaner logger: {e}")
            self.log_to_wandb = False
    
    def log_document_metrics(self, citations_found: Dict[str, List], citations_removed: Dict[str, List], 
                           citations_rejected: Dict[str, List], doc_metadata: Dict[str, str], 
                           figure_removal_stats: dict, enable_smart_validation: bool):
        """Loggt detaillierte per-document Metriken mit optimiertem Validation Logging"""
        if not self.wandb_initialized:
            return
            
        # Per Citation-Typ Metriken
        for citation_type in citations_found.keys():
            found_count = len(citations_found.get(citation_type, []))
            removed_count = len(citations_removed.get(citation_type, []))
            rejected_count = len(citations_rejected.get(citation_type, []))
            
            if found_count > 0:
                # Basic metrics für alle
                doc_metrics = {
                    f"{citation_type}/citations_found": found_count,
                    f"{citation_type}/citations_removed": removed_count,
                    f"{citation_type}/had_citations": 1 if removed_count > 0 else 0
                }
                
                # Validation metrics nur für validierte Types
                if self._has_validation(citation_type, enable_smart_validation):
                    doc_metrics.update({
                        f"{citation_type}/citations_rejected": rejected_count,
                        f"{citation_type}/validation_precision": removed_count / found_count if found_count > 0 else 0,
                        f"{citation_type}/false_positive_rate": rejected_count / found_count if found_count > 0 else 0
                    })
                
                wandb.log(doc_metrics)
        
        # Gesamt-Metriken nur wenn wir tatsächlich Validation haben
        total_found = sum(len(citations) for citations in citations_found.values())
        total_removed = sum(len(citations) for citations in citations_removed.values())
        total_rejected = sum(len(rejected) for rejected in citations_rejected.values())
        
        if total_found > 0:
            combined_metrics = {
                "combined/total_citations_found": total_found,
                "combined/total_citations_removed": total_removed,
                "combined/types_with_citations": sum(1 for citations in citations_removed.values() if len(citations) > 0)
            }
            
            # Validation metrics nur wenn Smart Validation aktiv
            if enable_smart_validation and total_rejected > 0:
                combined_metrics.update({
                    "combined/total_citations_rejected": total_rejected,
                    "combined/overall_precision": total_removed / total_found if total_found > 0 else 0,
                    "combined/false_positive_prevention_rate": total_rejected / total_found if total_found > 0 else 0
                })
            
            wandb.log(combined_metrics)
        
        # Figure line removal metrics
        if figure_removal_stats["lines_removed"] > 0:
            figure_metrics = {
                "figure_removal/lines_removed": figure_removal_stats["lines_removed"],
                "figure_removal/length_reduction": figure_removal_stats["length_reduction"],
                "figure_removal/had_figure_lines": 1
            }
            wandb.log(figure_metrics)
    
    def _has_validation(self, citation_type: str, enable_smart_validation: bool) -> bool:
        """Check if citation type has validation enabled"""
        if not enable_smart_validation:
            return False
        return citation_type in ["semicolon_blocks", "autor_jahr_text", "page_references", "isolated_numeric_citations"]
    
    def log_aggregated_stats(self, citation_stats: Dict, cleaning_stats: Dict, enable_smart_validation: bool):
        """Loggt optimierte aggregierte Stats"""
        if not self.wandb_initialized:
            return
        
        try:
            # Per Citation-Typ aggregierte Stats - nur relevante Metriken
            for citation_type, stats in citation_stats.items():
                if stats["total_citations_found"] == 0:
                    continue
                
                # Basic stats für alle
                agg_stats = {
                    f"{citation_type}_agg/docs_with_citations": stats["docs_with_citations"],
                    f"{citation_type}_agg/total_citations_found": stats["total_citations_found"],
                    f"{citation_type}_agg/total_citations_removed": stats["total_citations_removed"],
                    f"{citation_type}_agg/citation_rate": (
                        stats["docs_with_citations"] / cleaning_stats["docs_processed"]
                        if cleaning_stats["docs_processed"] > 0 else 0
                    )
                }
                
                # Validation stats nur für validierte Types
                if self._has_validation(citation_type, enable_smart_validation):
                    precision = (stats["total_citations_removed"] / stats["total_citations_found"] 
                               if stats["total_citations_found"] > 0 else 0)
                    agg_stats.update({
                        f"{citation_type}_agg/total_citations_rejected": stats["total_citations_rejected"],
                        f"{citation_type}_agg/validation_precision": precision,
                        f"{citation_type}_agg/false_positive_rate": (
                            stats["total_citations_rejected"] / stats["total_citations_found"]
                            if stats["total_citations_found"] > 0 else 0
                        )
                    })
                
                wandb.log(agg_stats)
            
            # Simplified combined stats
            if enable_smart_validation:
                total_found_all = sum(stats["total_citations_found"] for stats in citation_stats.values())
                if total_found_all > 0:
                    combined_stats = {
                        "combined_agg/docs_processed": cleaning_stats["docs_processed"],
                        "combined_agg/docs_with_any_citations": cleaning_stats["docs_with_any_citations"],
                        "combined_agg/total_citations_removed": cleaning_stats["total_citations_all_types"],
                        "combined_agg/overall_citation_rate": (
                            cleaning_stats["docs_with_any_citations"] / cleaning_stats["docs_processed"]
                            if cleaning_stats["docs_processed"] > 0 else 0
                        )
                    }
                    
                    # Validation stats nur wenn wir tatsächlich welche haben
                    if cleaning_stats["total_citations_rejected"] > 0:
                        combined_stats.update({
                            "combined_agg/total_citations_rejected": cleaning_stats["total_citations_rejected"],
                            "combined_agg/overall_validation_precision": (
                                cleaning_stats["total_citations_all_types"] / total_found_all
                            )
                        })
                    
                    wandb.log(combined_stats)
            
            log.info(f"📊 Citation stats: {cleaning_stats['docs_processed']} docs, "
                    f"{cleaning_stats['docs_with_any_citations']} with citations"
                    + (f", {cleaning_stats['total_citations_rejected']} rejected" 
                       if cleaning_stats['total_citations_rejected'] > 0 else ""))
            
        except Exception as e:
            log.warning(f"Failed to log optimized citation stats: {e}")
    
    def log_final_summary(self, citation_stats: Dict, cleaning_stats: Dict, enable_smart_validation: bool):
        """Loggt finale Summary mit optimiertem Validation Logging"""
        if not self.wandb_initialized:
            return
            
        # Final aggregated stats
        self.log_aggregated_stats(citation_stats, cleaning_stats, enable_smart_validation)
        
        # Smart Validation Summary nur wenn aktiviert
        if enable_smart_validation:
            self._log_validation_summary(citation_stats, cleaning_stats)
        
        # Citation-Type Comparison mit optimierten Metriken
        self._log_optimized_citation_comparison(citation_stats, enable_smart_validation)
        
        # Summary für jeden Citation-Typ
        self._log_citation_type_tables(citation_stats, enable_smart_validation)
        
        # Finale Summary Stats
        self._log_final_stats(citation_stats, cleaning_stats, enable_smart_validation)
        
        # Top Documents und Figure Line Removal Tables
        self._log_top_documents_tables(citation_stats, cleaning_stats)
    
    def _log_validation_summary(self, citation_stats: Dict, cleaning_stats: Dict):
        """Loggt detaillierte Smart Validation Summary"""
        try:
            # Per-Pattern Validation Stats
            validation_data = []
            for citation_type, stats in citation_stats.items():
                if stats["total_citations_found"] > 0:
                    precision = stats["total_citations_removed"] / stats["total_citations_found"]
                    fp_prevented = stats["total_citations_rejected"]
                    fp_rate = fp_prevented / stats["total_citations_found"]
                    
                    validation_data.append([
                        citation_type,
                        stats["total_citations_found"],
                        stats["total_citations_removed"],
                        fp_prevented,
                        f"{precision:.3f}",
                        f"{fp_rate:.3f}"
                    ])
            
            # Sortiere nach False Positive Rate (höchste zuerst)
            validation_data.sort(key=lambda x: float(x[5]), reverse=True)
            
            validation_table = wandb.Table(
                columns=["Citation Type", "Found", "Kept", "Rejected", "Precision", "FP Rate"],
                data=validation_data
            )
            wandb.log({"validation_summary/precision_by_type": validation_table})
            
            # Overall Validation Effectiveness
            total_found = sum(stats["total_citations_found"] for stats in citation_stats.values())
            total_rejected = cleaning_stats["total_citations_rejected"]
            
            if total_found > 0:
                effectiveness_stats = {
                    "validation_summary/total_citations_analyzed": total_found,
                    "validation_summary/total_false_positives_prevented": total_rejected,
                    "validation_summary/overall_false_positive_rate": total_rejected / total_found,
                    "validation_summary/validation_effectiveness": min(1.0, total_rejected / (total_found * 0.1))  # Assuming 10% baseline FP rate
                }
                wandb.log(effectiveness_stats)
            
            log.info(f"📊 Validation effectiveness: {total_rejected}/{total_found} potential false positives prevented "
                    f"({total_rejected/total_found:.1%} FP rate)")
            
        except Exception as e:
            log.warning(f"Failed to log validation summary: {e}")
    
    def _log_optimized_citation_comparison(self, citation_stats: Dict, enable_smart_validation: bool):
        """Loggt optimierte Citation-Type Vergleichs-Charts"""
        try:
            # Basic Citation Analysis für alle Types
            citation_analysis_data = []
            
            for citation_type, stats in citation_stats.items():
                if stats["total_citations_found"] > 0:
                    row = [
                        citation_type,
                        stats["total_citations_found"],
                        stats["total_citations_removed"],
                        stats["docs_with_citations"]
                    ]
                    
                    # Validation columns nur für validierte Types
                    if self._has_validation(citation_type, enable_smart_validation):
                        row.extend([
                            stats["total_citations_rejected"],
                            f"{(stats['total_citations_removed'] / stats['total_citations_found']):.3f}"
                        ])
                        row.append("✓")
                    else:
                        row.extend(["-", "-", "✗"])
                    
                    citation_analysis_data.append(row)
            
            # Sortiert nach Total Found Citations
            citation_analysis_data.sort(key=lambda x: x[1], reverse=True)
            
            # Citation Analysis Table
            analysis_table = wandb.Table(
                columns=["Citation Type", "Found", "Removed", "Docs", "Rejected", "Precision", "Validated"],
                data=citation_analysis_data
            )
            wandb.log({"tables/citation_analysis_optimized": analysis_table})
            
            # Risk Assessment nur für validierte Types
            if enable_smart_validation:
                risk_data = []
                for citation_type, stats in citation_stats.items():
                    if self._has_validation(citation_type, enable_smart_validation) and stats["total_citations_found"] > 0:
                        fp_rate = stats["total_citations_rejected"] / stats["total_citations_found"]
                        
                        risk_level = "High" if fp_rate > 0.3 else "Medium" if fp_rate > 0.1 else "Low"
                        
                        risk_data.append([
                            citation_type,
                            f"{fp_rate:.3f}",
                            risk_level,
                            stats["total_citations_found"],
                            stats["total_citations_rejected"]
                        ])
                
                if risk_data:
                    risk_data.sort(key=lambda x: float(x[1]), reverse=True)
                    
                    risk_table = wandb.Table(
                        columns=["Citation Type", "FP Rate", "Risk Level", "Total Found", "Rejected"],
                        data=risk_data
                    )
                    wandb.log({"tables/false_positive_risk_validated_only": risk_table})
            
            log.info(f"📊 Optimized citation comparison charts logged to W&B")
            
        except Exception as e:
            log.warning(f"Failed to log optimized citation comparison charts: {e}")
    
    def _log_citation_type_tables(self, citation_stats: Dict, enable_smart_validation: bool):
        """Loggt per-citation-type Tabellen und False Positive Samples"""
        for citation_type, stats in citation_stats.items():
            if stats["total_citations_found"] == 0:
                continue
                
            # Top Documents Table für alle Typen
            if stats["top_citation_docs"]:
                top_docs_data = []
                for citations, doc_id, title, _ in sorted(stats["top_citation_docs"], reverse=True):
                    top_docs_data.append([citations, doc_id, title[:80]])
                
                table = wandb.Table(
                    columns=["Citations", "Doc ID", "Title"],
                    data=top_docs_data
                )
                wandb.log({f"tables/top_{citation_type}_documents": table})
                
                log.info(f"📋 Top {citation_type} document: {top_docs_data[0][0]} citations")
            
            # False Positive Samples nur für validierte Types
            if (self._has_validation(citation_type, enable_smart_validation) and 
                stats["false_positive_samples"]):
                fp_data = []
                for sample in stats["false_positive_samples"]:
                    # Create context preview for W&B table
                    context_preview = ""
                    if "before_context" in sample and "after_context" in sample:
                        before = sample["before_context"][-100:] if len(sample["before_context"]) > 100 else sample["before_context"]
                        after = sample["after_context"][:100] if len(sample["after_context"]) > 100 else sample["after_context"]
                        context_preview = f"...{before}[{sample['match']}]{after}..."
                    
                    fp_data.append([
                        sample["match"][:60],
                        sample["doc_id"][:40],
                        sample["reason"][:100],
                        sample.get("position", "unknown"),
                        context_preview[:200]
                    ])
                
                fp_table = wandb.Table(
                    columns=["Rejected Match", "Doc ID", "Reason", "Position", "Context"],
                    data=fp_data
                )
                wandb.log({f"tables/{citation_type}_false_positives_rejected": fp_table})
                
                log.info(f"📋 {citation_type}: {len(stats['false_positive_samples'])} false positives prevented")
            
            # Distribution Histogram für alle
            if stats["citation_distribution"]:
                citation_counts = []
                for count, freq in stats["citation_distribution"].items():
                    citation_counts.extend([count] * freq)
                
                wandb.log({f"{citation_type}_distribution_histogram": wandb.Histogram(citation_counts)})
    
    def _log_final_stats(self, citation_stats: Dict, cleaning_stats: Dict, enable_smart_validation: bool):
        """Loggt finale Summary Stats"""
        total_found = sum(stats["total_citations_found"] for stats in citation_stats.values())
        final_stats = {
            "summary/total_docs_processed": cleaning_stats["docs_processed"],
            "summary/total_docs_with_any_citations": cleaning_stats["docs_with_any_citations"],
            "summary/total_citations_found": total_found,
            "summary/total_citations_removed": cleaning_stats["total_citations_all_types"],
            "summary/final_citation_rate": (
                cleaning_stats["docs_with_any_citations"] / cleaning_stats["docs_processed"]
                if cleaning_stats["docs_processed"] > 0 else 0
            ),
            "summary/total_length_reduction": cleaning_stats["total_length_reduction"],
            "summary/smart_validation_enabled": enable_smart_validation,
            
            # Figure line removal summary
            "summary/docs_with_figure_lines_removed": cleaning_stats["docs_with_figure_lines_removed"],
            "summary/total_figure_lines_removed": cleaning_stats["total_figure_lines_removed"],
            "summary/total_figure_line_length_reduction": cleaning_stats["total_figure_line_length_reduction"],
            "summary/figure_line_removal_rate": (
                cleaning_stats["docs_with_figure_lines_removed"] / cleaning_stats["docs_processed"]
                if cleaning_stats["docs_processed"] > 0 else 0
            ),
            
            # Citation limit exceeded summary
            "summary/docs_with_citation_limits_exceeded": cleaning_stats["docs_with_citation_limits_exceeded"],
            "summary/citation_limit_exceeded_rate": (
                cleaning_stats["docs_with_citation_limits_exceeded"] / cleaning_stats["docs_processed"]
                if cleaning_stats["docs_processed"] > 0 else 0
            ),
            
            # Language filtering summary
            "summary/docs_removed_by_language": cleaning_stats["docs_removed_by_language"],
            "summary/language_removal_rate": (
                cleaning_stats["docs_removed_by_language"] / cleaning_stats["docs_processed"]
                if cleaning_stats["docs_processed"] > 0 else 0
            ),
            "summary/average_fasttext_score": (
                cleaning_stats["total_fasttext_score"] / cleaning_stats["docs_processed"]
                if cleaning_stats["docs_processed"] > 0 else 1.0
            )
        }
        
        # Validation-spezifische finale Stats nur wenn relevant
        if enable_smart_validation and cleaning_stats["total_citations_rejected"] > 0:
            final_stats.update({
                "summary/total_citations_rejected": cleaning_stats["total_citations_rejected"],
                "summary/overall_precision": (
                    cleaning_stats["total_citations_all_types"] / total_found 
                    if total_found > 0 else 0
                ),
                "summary/false_positive_prevention_rate": (
                    cleaning_stats["total_citations_rejected"] / total_found 
                    if total_found > 0 else 0
                )
            })
        
        wandb.log(final_stats)
    
    def _log_top_documents_tables(self, citation_stats: Dict, cleaning_stats: Dict):
        """Loggt Top Documents Tabellen für Figure Line Removal und Combined"""
        
        # Figure Line Removal Samples Table
        if cleaning_stats["figure_line_removal_samples"]:
            figure_samples_data = []
            for sample in cleaning_stats["figure_line_removal_samples"]:
                figure_samples_data.append([
                    sample["doc_id"][:40],
                    sample["line_content"][:80],
                    sample["line_number"],
                    sample["reason"],
                    sample["length"]
                ])
            
            figure_samples_table = wandb.Table(
                columns=["Doc ID", "Removed Line Content", "Line Number", "Removal Reason", "Length"],
                data=figure_samples_data
            )
            wandb.log({"tables/figure_line_removal_samples": figure_samples_table})
            
            log.info(f"📋 Figure line removal: {len(cleaning_stats['figure_line_removal_samples'])} samples logged")
        
        # Top Figure Line Removal Documents
        if cleaning_stats["top_figure_line_removal_docs"]:
            top_figure_docs_data = []
            for lines_removed, doc_id, title, length_reduction in sorted(cleaning_stats["top_figure_line_removal_docs"], reverse=True):
                top_figure_docs_data.append([lines_removed, doc_id, title[:60], length_reduction])
            
            top_figure_table = wandb.Table(
                columns=["Lines Removed", "Doc ID", "Title", "Length Reduction"],
                data=top_figure_docs_data
            )
            wandb.log({"tables/top_figure_line_removal_documents": top_figure_table})
            
            log.info(f"📋 Top figure line removal document: {top_figure_docs_data[0][0]} lines removed")
        
        # Top Combined Reduction Documents (Citations + Figure Lines)
        if cleaning_stats["top_combined_reduction_docs"]:
            top_combined_docs_data = []
            for total_reduction, doc_id, title, combined_info in sorted(cleaning_stats["top_combined_reduction_docs"], reverse=True):
                top_combined_docs_data.append([total_reduction, doc_id, title[:60], combined_info])
            
            top_combined_table = wandb.Table(
                columns=["Total Reduction", "Doc ID", "Title", "Details (C=Citations, F=Figure Lines)"],
                data=top_combined_docs_data
            )
            wandb.log({"tables/top_combined_reduction_documents": top_combined_table})
            
            log.info(f"📋 Top combined reduction document: {top_combined_docs_data[0][0]} chars reduced")
        
        # Citation Limit Exceeded Documents Table
        if cleaning_stats["citation_limit_exceeded_samples"]:
            exceeded_docs_data = []
            for sample in cleaning_stats["citation_limit_exceeded_samples"]:
                exceeded_docs_data.append([
                    sample["doc_id"][:40],
                    sample["title"][:60], 
                    sample["citation_type"],
                    sample["count"],
                    sample["limit"],
                    sample["ratio"],
                    sample["doc_length"],
                    ", ".join(sample["first_matches"][:3])  # Erste 3 Matches
                ])
            
            exceeded_table = wandb.Table(
                columns=["Doc ID", "Title", "Citation Type", "Count", "Limit", "Ratio", "Doc Length", "Example Matches"],
                data=exceeded_docs_data
            )
            wandb.log({"tables/citation_limit_exceeded_documents": exceeded_table})
            
            log.info(f"📋 Citation limits exceeded: {len(cleaning_stats['citation_limit_exceeded_samples'])} documents logged")
        
        # Language filtering samples table
        if cleaning_stats["language_removal_samples"]:
            language_samples_data = []
            for sample in cleaning_stats["language_removal_samples"]:
                language_samples_data.append([
                    sample["doc_id"][:40],
                    sample["title"][:60],
                    f"{sample['fasttext_score']:.3f}",
                    sample["original_length"]
                ])
            
            language_table = wandb.Table(
                columns=["Doc ID", "Title", "FastText Score", "Original Length"],
                data=language_samples_data
            )
            wandb.log({"tables/language_filtered_documents": language_table})
            
 