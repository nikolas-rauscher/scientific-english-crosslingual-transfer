"""
Appendix Section Cleaner Module

Specialized cleaner for detecting and removing appendix/reference sections
that are harmful for T5 pretraining using character ratio analysis.

Extracted from MultiCitationCleaner to improve modularity and maintainability.
"""

import re
import logging
from typing import Dict, List, Tuple
from collections import defaultdict
from src.dataprep.text_cleaners import BaseTextCleaner

log = logging.getLogger(__name__)


class AppendixSectionCleaner(BaseTextCleaner):
    """
    Cleaner for appendix/reference sections using character ratio analysis
    
    Targets scientific tables, gene lists, acknowledgments, and reference sections
    that are harmful for T5 pretraining.
    """
    
    def __init__(self, debug_mode: bool = False):
        """
        Args:
            debug_mode: If True, add debug tags instead of removing sections
        """
        super().__init__(debug_mode)
        
        # PERFORMANCE OPTIMIZATION: Pre-compile all regex patterns
        self._compile_regex_patterns()
    
    def _compile_regex_patterns(self):
        """Pre-compile all regex patterns for optimal performance"""
        # Section indicator patterns
        section_patterns = [
            # Table indicators
            r'^table\s+\d+', r'^tab\s+\d+', r'^figure\s+\d+', r'^fig\s+\d+',
            
            # Scientific sections
            r'^acknowledgments?$', r'^acknowledgements?$', r'^references?$', 
            r'^bibliography$', r'^appendix', r'^supplementary',
            
            # Gene/protein lists
            r'^gene\s+symbol', r'^protein\s+name', r'^antibody\s+supplier',
            
            # Table content patterns
            r'^[A-Z][A-Z0-9]+\s+[A-Z][A-Z0-9]+\s+[A-Z][A-Z0-9]+',  # Column headers
        ]
        
        # Compile all section patterns with IGNORECASE
        self.compiled_section_patterns = [
            re.compile(pattern, re.IGNORECASE) for pattern in section_patterns
        ]
        
        # Compile frequently used patterns
        self.compiled_gene_pattern = re.compile(r'^[A-Z]{2,}')
        self.compiled_decimal_pattern = re.compile(r'\d+\.\d+')
        self.compiled_separator_pattern = re.compile(r'[:\t]')
        
        log.info(f"Compiled {len(self.compiled_section_patterns)} section patterns + 3 validation patterns")
    
    def clean(self, text: str, doc_id: str = None) -> Tuple[str, Dict]:
        """
        Detect and remove appendix/reference sections using character ratio analysis
        
        Args:
            text: Input text to clean
            doc_id: Document ID for logging (optional)
            
        Returns:
            Tuple of (cleaned_text, stats_dict)
        """
        lines = text.splitlines()
        cleaned_lines = []
        removed_sections = []
        current_section_lines = []
        in_appendix_section = False
        section_start_line = 0
        
        for i, line in enumerate(lines):
            stripped_line = line.strip()
            
            # Skip empty lines but don't reset section detection
            if not stripped_line:
                if in_appendix_section:
                    current_section_lines.append(line)
                else:
                    cleaned_lines.append(line)
                continue
            
            # OPTIMIZED: Use pre-compiled patterns (removed duplicate section_indicators list)
            
            is_section_start = any(pattern.search(stripped_line) 
                                 for pattern in self.compiled_section_patterns)
            
            if is_section_start and not in_appendix_section:
                in_appendix_section = True
                section_start_line = i + 1
                current_section_lines = [line]
                continue
            
            if in_appendix_section:
                current_section_lines.append(line)
                
                # Check if we should end the section (normal paragraph text returns)
                if len(current_section_lines) > 5:  # Only after minimum section length
                    # Calculate ratios for this line
                    line_stats = self._calculate_line_stats(stripped_line)
                    
                    # End section if we hit normal prose text
                    if (line_stats['word_count'] > 15 and 
                        line_stats['special_char_ratio'] < 0.15 and
                        line_stats['uppercase_ratio'] < 0.3 and
                        not self.compiled_gene_pattern.search(stripped_line) and  # Not gene names
                        not self.compiled_decimal_pattern.search(stripped_line) and    # Not version numbers
                        not self.compiled_separator_pattern.search(stripped_line)):         # Not tabular data
                        
                        # This looks like normal text, end the section
                        section_stats = self._analyze_section_stats(current_section_lines[:-1])  # Exclude current line
                        
                        if self._is_likely_appendix_section(section_stats):
                            removed_sections.append({
                                "start_line": section_start_line,
                                "end_line": i,
                                "lines_count": len(current_section_lines) - 1,
                                "total_length": sum(len(l) for l in current_section_lines[:-1]),
                                "section_type": section_stats.get('section_type', 'unknown'),
                                "confidence": section_stats.get('confidence', 0.0),
                                "sample_lines": current_section_lines[:3]  # First 3 lines as sample
                            })
                            
                            if self.debug_mode:
                                # Debug-Tag statt Entfernung
                                debug_tag = f"[DEBUG:appendix:{section_stats.get('section_type', 'unknown')}]"
                                cleaned_lines.append(debug_tag)
                            # Else: Section wird nicht hinzugefügt (entfernt)
                        else:
                            # Not an appendix, add back to cleaned text
                            cleaned_lines.extend(current_section_lines[:-1])
                        
                        # Reset and add current line as normal
                        in_appendix_section = False
                        current_section_lines = []
                        cleaned_lines.append(line)
                        continue
                
                # Continue collecting section lines
                continue
            
            # Normal line - add to cleaned text
            cleaned_lines.append(line)
        
        # Handle any remaining section at end of document
        if in_appendix_section and current_section_lines:
            section_stats = self._analyze_section_stats(current_section_lines)
            if self._is_likely_appendix_section(section_stats):
                removed_sections.append({
                    "start_line": section_start_line,
                    "end_line": len(lines),
                    "lines_count": len(current_section_lines),
                    "total_length": sum(len(l) for l in current_section_lines),
                    "section_type": section_stats.get('section_type', 'unknown'),
                    "confidence": section_stats.get('confidence', 0.0),
                    "sample_lines": current_section_lines[:3]
                })
                
                if self.debug_mode:
                    # Debug-Tag statt Entfernung
                    debug_tag = f"[DEBUG:appendix_end:{section_stats.get('section_type', 'unknown')}]"
                    cleaned_lines.append(debug_tag)
                # Else: Section wird nicht hinzugefügt (entfernt)
            else:
                cleaned_lines.extend(current_section_lines)
        
        # Statistics
        total_removed_lines = sum(section['lines_count'] for section in removed_sections)
        total_removed_length = sum(section['total_length'] for section in removed_sections)
        
        stats = {
            "sections_removed": len(removed_sections),
            "lines_removed": total_removed_lines,
            "length_reduction": total_removed_length if not self.debug_mode else 0,
            "removed_sections": removed_sections[:5],  # Store max 5 samples per document
            "doc_id": doc_id or "unknown"
        }
        
        return '\n'.join(cleaned_lines), stats
    
    def _calculate_line_stats(self, line: str) -> dict:
        """Calculate various statistics for a single line"""
        if not line:
            return {'word_count': 0, 'special_char_ratio': 0, 'uppercase_ratio': 0}
        
        words = line.split()
        word_count = len(words)
        
        # Special characters (non-alphanumeric except spaces)
        special_chars = len(re.findall(r'[^\w\s]', line))
        total_chars = len(line.replace(' ', ''))
        special_char_ratio = special_chars / total_chars if total_chars > 0 else 0
        
        # Uppercase letters
        uppercase_chars = sum(1 for c in line if c.isupper())
        letter_chars = sum(1 for c in line if c.isalpha())
        uppercase_ratio = uppercase_chars / letter_chars if letter_chars > 0 else 0
        
        return {
            'word_count': word_count,
            'special_char_ratio': special_char_ratio,
            'uppercase_ratio': uppercase_ratio,
            'total_length': len(line)
        }
    
    def _analyze_section_stats(self, section_lines: List[str]) -> dict:
        """Analyze statistics for an entire section to determine if it's an appendix"""
        if not section_lines:
            return {'confidence': 0.0, 'section_type': 'empty'}
        
        # Calculate aggregate statistics
        total_lines = len(section_lines)
        non_empty_lines = [line.strip() for line in section_lines if line.strip()]
        
        if not non_empty_lines:
            return {'confidence': 0.0, 'section_type': 'empty'}
        
        # Line-by-line analysis
        line_stats = [self._calculate_line_stats(line) for line in non_empty_lines]
        
        # Aggregate metrics
        avg_special_char_ratio = sum(s['special_char_ratio'] for s in line_stats) / len(line_stats)
        avg_uppercase_ratio = sum(s['uppercase_ratio'] for s in line_stats) / len(line_stats)
        avg_word_count = sum(s['word_count'] for s in line_stats) / len(line_stats)
        
        # Count specific patterns
        tabular_lines = sum(1 for line in non_empty_lines if '\t' in line or re.search(r'\s{3,}', line))
        gene_lines = sum(1 for line in non_empty_lines if re.search(r'^[A-Z0-9]{2,}\s+[A-Z0-9]{2,}', line))
        short_lines = sum(1 for s in line_stats if s['word_count'] < 5)
        numeric_lines = sum(1 for line in non_empty_lines if re.search(r'\d+\.\d+|\d+%', line))
        
        # Calculate confidence score
        confidence = 0.0
        section_type = "unknown"
        
        # CONSERVATIVE: Nur bei SEHR hohen Ratios als appendix klassifizieren
        
        # High special character ratio (tables, technical data) - erhöht von 0.2 auf 0.35
        if avg_special_char_ratio > 0.35:
            confidence += 0.3
            section_type = "technical_table"
        
        # High uppercase ratio (gene names, abbreviations) - bleibt bei 0.4
        if avg_uppercase_ratio > 0.4:
            confidence += 0.25
            section_type = "gene_list"
        
        # Many short lines (tabular data) - erhöht von 0.5 auf 0.7
        if short_lines / len(non_empty_lines) > 0.7:
            confidence += 0.2
            section_type = "tabular_data"
        
        # Tabular formatting - bleibt bei 0.3
        if tabular_lines / len(non_empty_lines) > 0.3:
            confidence += 0.3
            section_type = "formatted_table"
        
        # Gene name patterns - bleibt bei 0.3
        if gene_lines / len(non_empty_lines) > 0.3:
            confidence += 0.35
            section_type = "gene_table"
        
        # Many numeric values - erhöht von 0.5 auf 0.8 (Citations haben oft Jahre)
        if numeric_lines / len(non_empty_lines) > 0.8:
            confidence += 0.25
            section_type = "numeric_data"
        
        # Low average word count (not prose) - reduziert von 8 auf 5 (Citations haben längere Wörter)
        if avg_word_count < 5:
            confidence += 0.15
        
        return {
            'confidence': min(confidence, 1.0),
            'section_type': section_type,
            'total_lines': total_lines,
            'avg_special_char_ratio': avg_special_char_ratio,
            'avg_uppercase_ratio': avg_uppercase_ratio,
            'avg_word_count': avg_word_count,
            'tabular_lines_ratio': tabular_lines / len(non_empty_lines),
            'gene_lines_ratio': gene_lines / len(non_empty_lines)
        }
    
    def _is_likely_appendix_section(self, section_stats: dict) -> bool:
        """Determine if a section should be removed based on statistics"""
        confidence = section_stats.get('confidence', 0.0)
        
        # VERY Conservative threshold - erhöht von 0.6 auf 0.8 um wissenschaftlichen Text zu bewahren
        return confidence > 0.8 