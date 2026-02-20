"""
Citation Statistics Manager Module

Centralized statistics management for citation cleaning with comprehensive tracking
and analytics. Extracted from MultiCitationCleaner to improve modularity.

Handles:
- Per-citation-type statistics tracking
- Document-level metrics aggregation  
- Top documents and samples management
- False positive sample collection
- Overall cleaning statistics

Usage:
```python
from src.dataprep.citation_stats_manager import CitationStatsManager

stats_manager = CitationStatsManager(
    citation_patterns=citation_patterns,
    max_false_positive_samples=100,
    max_top_citation_docs=50
)

# Track citation results
stats_manager.track_citation_results(citation_type, found, removed, rejected, doc)

# Get current stats
citation_stats, cleaning_stats = stats_manager.get_stats()
```
"""

import heapq
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
from datatrove.data import Document


class CitationStatsManager:
    """Centralized statistics management for citation cleaning"""
    
    def __init__(
        self,
        citation_patterns: Dict[str, str],
        max_false_positive_samples: int = 100,
        max_top_citation_docs: int = 50,
        enable_smart_validation: bool = True
    ):
        """
        Args:
            citation_patterns: Dict of citation patterns for initialization
            max_false_positive_samples: Maximum false positive samples to store
            max_top_citation_docs: Maximum top documents to track
            enable_smart_validation: Whether smart validation is enabled
        """
        self.max_false_positive_samples = max_false_positive_samples
        self.max_top_citation_docs = max_top_citation_docs
        self.enable_smart_validation = enable_smart_validation
        
        # Initialize stats for each citation type
        self.citation_stats = {}
        for citation_type in citation_patterns.keys():
            self.citation_stats[citation_type] = {
                "docs_with_citations": 0,
                "total_citations_removed": 0,
                "total_citations_found": 0,
                "total_citations_rejected": 0,
                "total_length_reduction": 0,
                "citation_distribution": defaultdict(int),
                "top_citation_docs": [],
                "false_positive_samples": []
            }
        
        # Overall cleaning stats
        self.cleaning_stats = {
            "docs_processed": 0,
            "docs_with_any_citations": 0,
            "total_citations_all_types": 0,
            "total_citations_rejected": 0,
            "total_length_reduction": 0,
            "total_word_reduction": 0,
            "smart_validation_enabled": self.enable_smart_validation,
            
            # Figure-only line removal stats
            "docs_with_figure_lines_removed": 0,
            "total_figure_lines_removed": 0,
            "total_figure_line_length_reduction": 0,
            "figure_line_removal_samples": [],
            
            # Appendix section removal stats
            "docs_with_appendix_sections_removed": 0,
            "total_appendix_sections_removed": 0,
            "total_appendix_lines_removed": 0,
            "total_appendix_length_reduction": 0,
            "appendix_removal_samples": [],
            
            # Short line removal stats
            "docs_with_short_lines_removed": 0,
            "total_short_lines_removed": 0,
            "total_short_line_length_reduction": 0,
            "short_line_removal_samples": [],
            
            # Top Documents tracking
            "top_figure_line_removal_docs": [],
            "top_appendix_removal_docs": [],
            "top_short_line_removal_docs": [],
            "top_combined_reduction_docs": [],
            "language_cleaning_documents": [],
            
            # Language filtering stats (simple)
            "docs_removed_by_language": 0,
            "total_fasttext_score": 0.0,
            "language_removal_samples": [],
        }
    
    def track_citation_results(
        self,
        citation_type: str,
        citations_found: List[str],
        citations_removed: List[str], 
        citations_rejected: List[Dict],
        doc: Document
    ):
        """Track citation results for a specific type and document"""
        total_found = len(citations_found)
        total_removed = len(citations_removed)
        total_rejected = len(citations_rejected)
        
        # Update citation type stats
        self.citation_stats[citation_type]["total_citations_found"] += total_found
        self.citation_stats[citation_type]["total_citations_removed"] += total_removed
        self.citation_stats[citation_type]["total_citations_rejected"] += total_rejected
        
        # Track false positive samples
        if self.enable_smart_validation and total_rejected > 0:
            for rejected_item in citations_rejected:
                if len(self.citation_stats[citation_type]["false_positive_samples"]) < self.max_false_positive_samples:
                    # Extract context around the rejection
                    sample = {
                        "match": rejected_item["match"],
                        "doc_id": str(doc.id) if doc.id else "unknown",
                        "reason": rejected_item["reason"],
                        "confidence": 0.0,  # Simple validator doesn't provide confidence
                        "position": f"{rejected_item['position'][0]}-{rejected_item['position'][1]}",
                    }
                    
                    # Add context if available
                    if "before_context" in rejected_item:
                        sample["before_context"] = rejected_item["before_context"]
                    if "after_context" in rejected_item:
                        sample["after_context"] = rejected_item["after_context"]
                    
                    self.citation_stats[citation_type]["false_positive_samples"].append(sample)
        
        # Track document-level stats if citations were removed
        if total_removed > 0:
            self.citation_stats[citation_type]["docs_with_citations"] += 1
            self.citation_stats[citation_type]["citation_distribution"][total_removed] += 1
            
            # Track top documents
            doc_entry = (
                total_removed,
                str(doc.id) if doc.id else "unknown",
                str(doc.metadata.get("title", ""))[:50] if doc.metadata else "",
                ""
            )
            
            if len(self.citation_stats[citation_type]["top_citation_docs"]) < self.max_top_citation_docs:
                heapq.heappush(self.citation_stats[citation_type]["top_citation_docs"], doc_entry)
            elif total_removed > self.citation_stats[citation_type]["top_citation_docs"][0][0]:
                heapq.heapreplace(self.citation_stats[citation_type]["top_citation_docs"], doc_entry)
    
    def track_document_processing(
        self,
        doc: Document,
        total_citations_removed: int,
        total_citations_rejected: int,
        length_reduction: int = 0,
        word_reduction: int = 0
    ):
        """Track overall document processing stats"""
        self.cleaning_stats["docs_processed"] += 1
        
        if total_citations_removed > 0:
            self.cleaning_stats["docs_with_any_citations"] += 1
            self.cleaning_stats["total_citations_all_types"] += total_citations_removed
        
        if total_citations_rejected > 0:
            self.cleaning_stats["total_citations_rejected"] += total_citations_rejected
        
        if length_reduction > 0:
            self.cleaning_stats["total_length_reduction"] += length_reduction
            self.cleaning_stats["total_word_reduction"] += word_reduction
    
    def track_figure_line_removal(
        self,
        doc: Document,
        figure_removal_stats: Dict[str, Any]
    ):
        """Track figure line removal statistics"""
        if figure_removal_stats["lines_removed"] > 0:
            self.cleaning_stats["docs_with_figure_lines_removed"] += 1
            self.cleaning_stats["total_figure_lines_removed"] += figure_removal_stats["lines_removed"]
            self.cleaning_stats["total_figure_line_length_reduction"] += figure_removal_stats["length_reduction"]
            
            # Store samples
            if len(self.cleaning_stats["figure_line_removal_samples"]) < self.max_false_positive_samples:
                for removed_line in figure_removal_stats["removed_lines"]:
                    if len(self.cleaning_stats["figure_line_removal_samples"]) < self.max_false_positive_samples:
                        sample = {
                            "doc_id": figure_removal_stats["doc_id"],
                            "line_content": removed_line["line_content"],
                            "line_number": removed_line["line_number"],
                            "reason": removed_line["reason"],
                            "length": removed_line["length"]
                        }
                        self.cleaning_stats["figure_line_removal_samples"].append(sample)
            
            # Track top documents
            doc_title = str(doc.metadata.get("title", ""))[:50] if doc.metadata else ""
            figure_doc_entry = (
                figure_removal_stats["lines_removed"],
                figure_removal_stats["doc_id"],
                doc_title,
                figure_removal_stats["length_reduction"]
            )
            
            if len(self.cleaning_stats["top_figure_line_removal_docs"]) < self.max_top_citation_docs:
                heapq.heappush(self.cleaning_stats["top_figure_line_removal_docs"], figure_doc_entry)
            elif figure_removal_stats["lines_removed"] > self.cleaning_stats["top_figure_line_removal_docs"][0][0]:
                heapq.heapreplace(self.cleaning_stats["top_figure_line_removal_docs"], figure_doc_entry)
    
    def track_appendix_section_removal(
        self,
        doc: Document,
        appendix_removal_stats: Dict[str, Any]
    ):
        """Track appendix section removal statistics"""
        if appendix_removal_stats["sections_removed"] > 0:
            self.cleaning_stats["docs_with_appendix_sections_removed"] += 1
            self.cleaning_stats["total_appendix_sections_removed"] += appendix_removal_stats["sections_removed"]
            self.cleaning_stats["total_appendix_lines_removed"] += appendix_removal_stats["lines_removed"]
            self.cleaning_stats["total_appendix_length_reduction"] += appendix_removal_stats["length_reduction"]
            
            # Store samples
            if len(self.cleaning_stats["appendix_removal_samples"]) < self.max_false_positive_samples:
                for removed_section in appendix_removal_stats["removed_sections"]:
                    if len(self.cleaning_stats["appendix_removal_samples"]) < self.max_false_positive_samples:
                        sample = {
                            "doc_id": appendix_removal_stats["doc_id"],
                            "section_type": removed_section["section_type"],
                            "lines_count": removed_section["lines_count"],
                            "start_line": removed_section["start_line"],
                            "confidence": removed_section["confidence"],
                            "sample_lines": removed_section["sample_lines"]
                        }
                        self.cleaning_stats["appendix_removal_samples"].append(sample)
            
            # Track top documents
            doc_title = str(doc.metadata.get("title", ""))[:50] if doc.metadata else ""
            appendix_doc_entry = (
                appendix_removal_stats["sections_removed"],
                appendix_removal_stats["doc_id"],
                doc_title,
                appendix_removal_stats["length_reduction"]
            )
            
            if len(self.cleaning_stats["top_appendix_removal_docs"]) < self.max_top_citation_docs:
                heapq.heappush(self.cleaning_stats["top_appendix_removal_docs"], appendix_doc_entry)
            elif appendix_removal_stats["sections_removed"] > self.cleaning_stats["top_appendix_removal_docs"][0][0]:
                heapq.heapreplace(self.cleaning_stats["top_appendix_removal_docs"], appendix_doc_entry)
    
    def track_short_line_removal(
        self,
        doc: Document,
        short_line_removal_stats: Dict[str, Any]
    ):
        """Track short line removal statistics"""
        if short_line_removal_stats["total_lines_removed"] > 0:
            self.cleaning_stats["docs_with_short_lines_removed"] += 1
            self.cleaning_stats["total_short_lines_removed"] += short_line_removal_stats["total_lines_removed"]
            self.cleaning_stats["total_short_line_length_reduction"] += short_line_removal_stats["total_length_reduction"]
            
            # Store samples
            if len(self.cleaning_stats["short_line_removal_samples"]) < self.max_false_positive_samples:
                for removed_line in short_line_removal_stats["removed_lines"]:
                    if len(self.cleaning_stats["short_line_removal_samples"]) < self.max_false_positive_samples:
                        sample = {
                            "doc_id": short_line_removal_stats["doc_id"],
                            "line_content": removed_line["line_content"],
                            "line_number": removed_line["line_number"],
                            "word_count": removed_line.get("word_count", 0),
                            "reason": removed_line["reason"],
                            "length": removed_line["length"]
                        }
                        self.cleaning_stats["short_line_removal_samples"].append(sample)
            
            # Track top documents
            doc_title = str(doc.metadata.get("title", ""))[:50] if doc.metadata else ""
            short_line_doc_entry = (
                short_line_removal_stats["total_lines_removed"],
                short_line_removal_stats["doc_id"],
                doc_title,
                short_line_removal_stats["total_length_reduction"]
            )
            
            if len(self.cleaning_stats["top_short_line_removal_docs"]) < self.max_top_citation_docs:
                heapq.heappush(self.cleaning_stats["top_short_line_removal_docs"], short_line_doc_entry)
            elif short_line_removal_stats["total_lines_removed"] > self.cleaning_stats["top_short_line_removal_docs"][0][0]:
                heapq.heapreplace(self.cleaning_stats["top_short_line_removal_docs"], short_line_doc_entry)
    
    def track_language_cleaning(self, doc: Document, language_stats: Dict[str, Any]):
        """Track simple language filtering stats"""
        fasttext_score = language_stats.get("fasttext_score", 1.0)
        self.cleaning_stats["total_fasttext_score"] += fasttext_score
        
        # Track docs removed due to low FastText score
        if language_stats.get("document_removed", False):
            self.cleaning_stats["docs_removed_by_language"] += 1
            
            # Store sample for W&B (only near threshold examples)
            if len(self.cleaning_stats["language_removal_samples"]) < self.max_false_positive_samples:
                sample = {
                    "doc_id": str(doc.id),
                    "title": str(doc.metadata.get("title", ""))[:80],
                    "fasttext_score": fasttext_score,
                    "original_length": language_stats.get("original_length", 0)
                }
                self.cleaning_stats["language_removal_samples"].append(sample)

    def track_combined_reduction(
        self,
        doc: Document,
        total_citations_removed: int,
        figure_removal_stats: Dict[str, Any],
        appendix_removal_stats: Dict[str, Any],
        short_line_removal_stats: Dict[str, Any],
        total_combined_reduction: int
    ):
        """Track combined reduction from all cleaning methods"""
        if (total_combined_reduction > 0 or total_citations_removed > 0 or 
            figure_removal_stats.get("lines_removed", 0) > 0 or
            appendix_removal_stats.get("sections_removed", 0) > 0 or
            short_line_removal_stats.get("total_lines_removed", 0) > 0):
            
            doc_title = str(doc.metadata.get("title", ""))[:50] if doc.metadata else ""
            combined_info = (f"C:{total_citations_removed},"
                           f"F:{figure_removal_stats.get('lines_removed', 0)},"
                           f"A:{appendix_removal_stats.get('sections_removed', 0)},"
                           f"S:{short_line_removal_stats.get('total_lines_removed', 0)}")
            
            combined_doc_entry = (
                total_combined_reduction,
                str(doc.id) if doc.id else "unknown",
                doc_title,
                combined_info
            )
            
            if len(self.cleaning_stats["top_combined_reduction_docs"]) < self.max_top_citation_docs:
                heapq.heappush(self.cleaning_stats["top_combined_reduction_docs"], combined_doc_entry)
            elif total_combined_reduction > self.cleaning_stats["top_combined_reduction_docs"][0][0]:
                heapq.heapreplace(self.cleaning_stats["top_combined_reduction_docs"], combined_doc_entry)
    
    def track_citation_validation(self, citation_type: str, total_found: int, total_validated: int, total_rejected: int):
        """Track citation validation statistics for a specific citation type"""
        if citation_type not in self.citation_stats:
            return
        pass
    
    def get_stats(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Get current citation and cleaning statistics"""
        return self.citation_stats, self.cleaning_stats
    
    def reset_stats(self):
        """Reset all statistics (useful for testing)"""
        for citation_type in self.citation_stats.keys():
            self.citation_stats[citation_type] = {
                "docs_with_citations": 0,
                "total_citations_removed": 0,
                "total_citations_found": 0,
                "total_citations_rejected": 0,
                "total_length_reduction": 0,
                "citation_distribution": defaultdict(int),
                "top_citation_docs": [],
                "false_positive_samples": []
            }
        
        # Reset all cleaning stats counters but keep configuration
        self.cleaning_stats.update({
            "docs_processed": 0,
            "docs_with_any_citations": 0,
            "total_citations_all_types": 0,
            "total_citations_rejected": 0,
            "total_length_reduction": 0,
            "total_word_reduction": 0,
            "docs_with_figure_lines_removed": 0,
            "total_figure_lines_removed": 0,
            "total_figure_line_length_reduction": 0,
            "figure_line_removal_samples": [],
            "docs_with_appendix_sections_removed": 0,
            "total_appendix_sections_removed": 0,
            "total_appendix_lines_removed": 0,
            "total_appendix_length_reduction": 0,
            "appendix_removal_samples": [],
            "docs_with_short_lines_removed": 0,
            "total_short_lines_removed": 0,
            "total_short_line_length_reduction": 0,
            "short_line_removal_samples": [],
            "top_figure_line_removal_docs": [],
            "top_appendix_removal_docs": [],
            "top_short_line_removal_docs": [],
            "top_combined_reduction_docs": [],
            "language_cleaning_documents": [],
        }) 