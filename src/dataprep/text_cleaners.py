"""
Specialized Text Cleaners für DataTrove
"""

import re
from typing import Dict, List, Tuple, Set
from abc import ABC, abstractmethod


class BaseTextCleaner(ABC):
    """Base class for all text cleaners"""
    
    def __init__(self, debug_mode: bool = False):
        self.debug_mode = debug_mode
    
    @abstractmethod
    def clean(self, text: str, doc_id: str = None) -> Tuple[str, Dict]:
        """Clean text and return cleaned text with stats"""
        pass


class RepetitionCleaner(BaseTextCleaner):
    """
    Cleaner für wiederholte Zeilen (OCR-Artefakte)
    """
    
    def __init__(self, max_recent_lines: int = 5, debug_mode: bool = False):
        """
        Args:
            max_recent_lines: Anzahl der letzten Zeilen für Wiederholungserkennung
        """
        super().__init__(debug_mode)
        self.max_recent_lines = max_recent_lines
    
    def clean(self, text: str, doc_id: str = None) -> Tuple[str, Dict]:
        """Remove repeated lines that are likely OCR artifacts"""
        lines = text.splitlines()
        cleaned_lines = []
        removed_lines = []
        recent_lines = []
        
        for i, line in enumerate(lines):
            stripped_line = line.strip()
            
            # Keep empty lines
            if not stripped_line:
                cleaned_lines.append(line)
                continue
            
            # Check for repetition
            if stripped_line in recent_lines:
                # Skip debug tags - they can create false positives when different items get same tag
                if stripped_line.startswith("[DEBUG:") and stripped_line.endswith("]"):
                    cleaned_lines.append(line)
                    continue
                    
                removed_lines.append({
                    "line_content": stripped_line[:100],
                    "line_number": i + 1,
                    "length": len(stripped_line),
                    "reason": "repeated_line"
                })
                
                if self.debug_mode:
                    # Debug-Tag statt Entfernung
                    debug_tag = f"[DEBUG:repetition:repeated_line]"
                    cleaned_lines.append(debug_tag)
                else:
                    # Line wird nicht hinzugefügt (entfernt)
                    continue
            else:
                # Update tracking and keep line
                recent_lines.append(stripped_line)
                if len(recent_lines) > self.max_recent_lines:
                    recent_lines.pop(0)
                
                cleaned_lines.append(line)
        
        stats = {
            "lines_removed": len(removed_lines),
            "length_reduction": sum(item["length"] for item in removed_lines) if not self.debug_mode else 0,
            "removed_lines": removed_lines[:10],
            "doc_id": doc_id or "unknown"
        }
        
        return '\n'.join(cleaned_lines), stats


class TabularDataCleaner(BaseTextCleaner):
    """
    Cleaner für Tabellendaten mit übermäßigen Leerzeichen
    """
    
    def __init__(self, min_space_sequences: int = 2, space_ratio_threshold: float = 0.4, debug_mode: bool = False):
        """
        Args:
            min_space_sequences: Mindestanzahl von 3+-Leerzeichen-Sequenzen
            space_ratio_threshold: Schwellenwert für Leerzeichen-Anteil
        """
        super().__init__(debug_mode)
        self.min_space_sequences = min_space_sequences
        self.space_ratio_threshold = space_ratio_threshold
        self._compile_regex_patterns()
    
    def _compile_regex_patterns(self):
        """Pre-compile regex patterns for better performance"""
        self.multiple_spaces_pattern = re.compile(r'\s{3,}')
    
    def clean(self, text: str, doc_id: str = None) -> Tuple[str, Dict]:
        """Remove lines with excessive spacing indicating tabular data"""
        lines = text.splitlines()
        cleaned_lines = []
        removed_lines = []
        
        for i, line in enumerate(lines):
            stripped_line = line.strip()
            
            # Keep empty lines
            if not stripped_line:
                cleaned_lines.append(line)
                continue
            
            should_remove, reason = self._is_tabular_data(stripped_line)
            
            if should_remove:
                removed_lines.append({
                    "line_content": stripped_line[:100],
                    "line_number": i + 1,
                    "length": len(stripped_line),
                    "reason": reason
                })
                
                if self.debug_mode:
                    # Debug-Tag statt Entfernung
                    debug_tag = f"[DEBUG:tabular:{reason}]"
                    cleaned_lines.append(debug_tag)
                else:
                    # Line wird nicht hinzugefügt (entfernt)
                    continue
            else:
                cleaned_lines.append(line)
        
        stats = {
            "lines_removed": len(removed_lines),
            "length_reduction": sum(item["length"] for item in removed_lines) if not self.debug_mode else 0,
            "removed_lines": removed_lines[:10],
            "doc_id": doc_id or "unknown"
        }
        
        return '\n'.join(cleaned_lines), stats
    
    def _is_tabular_data(self, line: str) -> Tuple[bool, str]:
        """Check if line represents tabular data - DISABLED"""
        # This method is disabled - only consecutive tab logic is used
        return False, ""


class ShortLineCleaner(BaseTextCleaner):
    """
    Cleaner für isolierte kurze Zeilen (1-3 Wörter)
    """
    
    def __init__(self, debug_mode: bool = False):
        super().__init__(debug_mode)
        self._compile_regex_patterns()
    
    def _compile_regex_patterns(self):
        """Compile useful specific patterns"""
        self.keywords_pattern = re.compile(r'^KEYWORDS?:\s+', re.IGNORECASE)
        self.isolated_table_pattern = re.compile(r'^TABLE\s+\d+$', re.IGNORECASE)
        self.isolated_figure_pattern = re.compile(r'^FIGURE\s+\d+$', re.IGNORECASE)
        self.isolated_chapter_pattern = re.compile(r'^CHAPTER\s+\d+$', re.IGNORECASE)
        self.page_number_pattern = re.compile(r'^p\.?\s*\d+$', re.IGNORECASE)
        self.junk_pattern = re.compile(r'^[\d\s\.\,\;\-\(\)]+$')
    
    def clean(self, text: str, doc_id: str = None) -> Tuple[str, Dict]:
        """Remove isolated short lines that break text flow"""
        lines = text.splitlines()
        cleaned_lines = []
        removed_lines = []
        
        for i, line in enumerate(lines):
            stripped_line = line.strip()
            
            # Keep empty lines
            if not stripped_line:
                cleaned_lines.append(line)
                continue
            
            words = stripped_line.split()
            word_count = len(words)
            
            should_remove, reason = self._should_remove_short_line(stripped_line, words, word_count)
            
            if should_remove:
                removed_lines.append({
                    "line_content": stripped_line[:100],
                    "line_number": i + 1,
                    "word_count": word_count,
                    "length": len(stripped_line),
                    "reason": reason
                })
                
                if self.debug_mode:
                    # Debug-Tag statt Entfernung
                    debug_tag = f"[DEBUG:short_line:{reason}]"
                    cleaned_lines.append(debug_tag)
                else:
                    # Line wird nicht hinzugefügt (entfernt)
                    continue
            else:
                cleaned_lines.append(line)
        
        stats = {
            "lines_removed": len(removed_lines),
            "length_reduction": sum(item["length"] for item in removed_lines) if not self.debug_mode else 0,
            "removed_lines": removed_lines[:10],
            "doc_id": doc_id or "unknown"
        }
        
        return '\n'.join(cleaned_lines), stats
    
    def _should_remove_short_line(self, line: str, words: List[str], word_count: int) -> Tuple[bool, str]:
        """Smart removal: specific patterns + general fallback"""
        stripped = line.strip()
        
        # Keep longer lines (>15 chars) - likely real content
        if len(stripped) > 15:
            return False, ""
        
        # Remove specific patterns (even if alphabetic)
        if self.keywords_pattern.match(stripped):
            return True, "keywords_metadata"
        
        if self.isolated_table_pattern.match(stripped):
            return True, "isolated_table_reference"
            
        if self.isolated_figure_pattern.match(stripped):
            return True, "isolated_figure_reference"
            
        if self.isolated_chapter_pattern.match(stripped):
            return True, "isolated_chapter_reference"
        
        if self.page_number_pattern.match(stripped):
            return True, "page_number"
            
        if self.junk_pattern.match(stripped):
            return True, "numeric_junk"
        
        # Keep alphabetic content (headers like "Introduction", "METHODS") 
        if re.match(r'^[A-Za-z\s]+$', stripped):
            return False, ""
        
        # Conservative: keep everything else
        return False, ""


class CopyrightCleaner(BaseTextCleaner):
    """
    Cleaner für Copyright-Zeilen (basierend auf 170k Rohdaten-Analyse)
    ~52k Dokumente betroffen - größter Impact nach URLs
    """
    
    def __init__(self, debug_mode: bool = False):
        super().__init__(debug_mode)
        self.copyright_patterns = [
            r'.*©.*',
            r'.*copyright.*', 
            r'.*\(c\).*',
            r'.*all rights reserved.*'
        ]
        self._compile_regex_patterns()
    
    def _compile_regex_patterns(self):
        """Pre-compile regex patterns for better performance"""
        self.compiled_copyright_patterns = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.copyright_patterns
        ]
    
    def clean(self, text: str, doc_id: str = None) -> Tuple[str, Dict]:
        """Remove copyright lines completely"""
        lines = text.splitlines()
        cleaned_lines = []
        removed_lines = []
        
        for i, line in enumerate(lines):
            stripped_line = line.strip()
            
            # Keep empty lines
            if not stripped_line:
                cleaned_lines.append(line)
                continue
            
            # Check for copyright patterns
            is_copyright_line = any(pattern.search(stripped_line) 
                                  for pattern in self.compiled_copyright_patterns)
            
            if is_copyright_line:
                removed_lines.append({
                    "line_content": stripped_line[:100],
                    "line_number": i + 1,
                    "length": len(stripped_line),
                    "reason": "copyright_line"
                })
                
                if self.debug_mode:
                    # Debug-Tag statt Entfernung
                    debug_tag = f"[DEBUG:copyright:copyright_line]"
                    cleaned_lines.append(debug_tag)
                else:
                    # Ganze Zeile wird entfernt
                    continue
            else:
                cleaned_lines.append(line)
        
        stats = {
            "lines_removed": len(removed_lines),
            "length_reduction": sum(item["length"] for item in removed_lines) if not self.debug_mode else 0,
            "removed_lines": removed_lines[:10],
            "doc_id": doc_id or "unknown"
        }
        
        return '\n'.join(cleaned_lines), stats


class MetadataCleaner(BaseTextCleaner):
    """
    Cleaner für Metadata-Zeilen (Keywords, technische Codes)
    """
    
    def __init__(self, debug_mode: bool = False):
        super().__init__(debug_mode)
        self._compile_regex_patterns()
    
    def _compile_regex_patterns(self):
        """Pre-compile regex patterns for better performance"""
        self.keywords_metadata_pattern = re.compile(r'^[Kk]eywords?:\s+')
        self.technical_code_start_pattern = re.compile(r'^[A-Z0-9\-]{2,}\s+[A-Z0-9\-]{2,}')
        self.all_caps_pattern = re.compile(r'^[A-Z0-9\s\-\.]+$')
        self.lowercase_pattern = re.compile(r'[a-z]')
    
    def clean(self, text: str, doc_id: str = None) -> Tuple[str, Dict]:
        """Remove metadata lines like keywords and technical codes"""
        lines = text.splitlines()
        cleaned_lines = []
        removed_lines = []
        
        for i, line in enumerate(lines):
            stripped_line = line.strip()
            
            # Keep empty lines
            if not stripped_line:
                cleaned_lines.append(line)
                continue
            
            should_remove, reason = self._is_metadata_line(stripped_line)
            
            if should_remove:
                removed_lines.append({
                    "line_content": stripped_line[:100],
                    "line_number": i + 1,
                    "length": len(stripped_line),
                    "reason": reason
                })
                
                if self.debug_mode:
                    # Debug-Tag statt Entfernung
                    debug_tag = f"[DEBUG:metadata:{reason}]"
                    cleaned_lines.append(debug_tag)
                else:
                    # Line wird nicht hinzugefügt (entfernt)
                    continue
            else:
                cleaned_lines.append(line)
        
        stats = {
            "lines_removed": len(removed_lines),
            "length_reduction": sum(item["length"] for item in removed_lines) if not self.debug_mode else 0,
            "removed_lines": removed_lines[:10],
            "doc_id": doc_id or "unknown"
        }
        
        return '\n'.join(cleaned_lines), stats
    
    def _is_metadata_line(self, line: str) -> Tuple[bool, str]:
        """Check if line is metadata that should be removed"""
        # Keywords lines
        if self.keywords_metadata_pattern.match(line):
            return True, "keywords_metadata"
        
        # Technical codes (only short ones, not drug names like GEFITINIB)
        if (len(line.split()) <= 8 and
            self.technical_code_start_pattern.search(line) or
            (self.all_caps_pattern.match(line) and
             not self.lowercase_pattern.search(line) and
             len(line.replace(' ', '')) < 8)):  # Only short codes
            return True, "technical_code_line"
        
        return False, ""


class ComprehensiveLineCleaner:
    """
    Orchestrator der alle spezialisierten Cleaner kombiniert
    """
    
    def __init__(self, debug_mode: bool = False):
        self.repetition_cleaner = RepetitionCleaner(max_recent_lines=5, debug_mode=debug_mode)
        self.tabular_cleaner = TabularDataCleaner(min_space_sequences=2, space_ratio_threshold=0.4, debug_mode=debug_mode)
        self.short_line_cleaner = ShortLineCleaner(debug_mode=debug_mode)
        self.metadata_cleaner = MetadataCleaner(debug_mode=debug_mode)
        self.copyright_cleaner = CopyrightCleaner(debug_mode=debug_mode)  # NEUE Copyright-Cleaner
        self.debug_mode = debug_mode
    
    def clean_lines(self, text: str, doc_id: str = None) -> Tuple[str, Dict]:
        """
        Apply all line cleaning steps using optimized single-pass approach
        
        Returns:
            Tuple of (cleaned_text, combined_stats)
        """
        # Use optimized single-pass implementation
        return self.clean_lines_optimized(text, doc_id)
    
    def clean_lines_optimized(self, text: str, doc_id: str = None) -> Tuple[str, Dict]:
        """OPTIMIZED: Single-pass line cleaning with smart consecutive tab detection"""
        lines = text.splitlines()
        cleaned_lines = []
        
        # Stats tracking for each cleaner type
        stats_by_type = {
            "repetition": {"lines_removed": 0, "length_reduction": 0, "removed_lines": []},
            "tabular": {"lines_removed": 0, "length_reduction": 0, "removed_lines": []},
            "short_lines": {"lines_removed": 0, "length_reduction": 0, "removed_lines": []},
            "metadata": {"lines_removed": 0, "length_reduction": 0, "removed_lines": []},
            "copyright": {"lines_removed": 0, "length_reduction": 0, "removed_lines": []}
        }
        
        recent_lines = []  # For repetition detection
        
        # SMART TABLE DETECTION: Check for consecutive lines with tabs
        table_line_indices = set()
        for i in range(len(lines) - 1):
            current_line = lines[i]  # Don't strip - keep original formatting
            next_line = lines[i + 1]  # Don't strip - keep original formatting
            
            # If current line AND next line both contain multiple tabs -> likely table
            if ('\t' in current_line and current_line.count('\t') >= 2 and 
                '\t' in next_line and next_line.count('\t') >= 2):
                table_line_indices.add(i)
                table_line_indices.add(i + 1)
        
        for i, line in enumerate(lines):
            stripped_line = line.strip()
            
            # Skip empty lines
            if not stripped_line:
                cleaned_lines.append(line)
                continue
                
            removal_reason = None
            cleaner_type = None
            
            # FUSED LOGIC: Apply all 5 cleaners in priority order
            
            # 1. Check repetition (use existing logic)
            if self._should_remove_repetition_optimized(stripped_line, recent_lines):
                removal_reason = "repeated_line"
                cleaner_type = "repetition"
            
            # 2. Check tabular data (ONLY consecutive tab detection)
            elif i in table_line_indices:
                removal_reason = "consecutive_tab_lines"
                cleaner_type = "tabular"
                
            # 3. Check short lines (use existing logic)
            elif self._should_remove_short_line_optimized(stripped_line):
                removal_reason = self._get_short_line_reason_optimized(stripped_line)
                cleaner_type = "short_lines"
                
            # 4. Check metadata (use existing logic)
            elif self._should_remove_metadata_optimized(stripped_line):
                removal_reason = self._get_metadata_reason_optimized(stripped_line)
                cleaner_type = "metadata"
                
            # 5. Check copyright (use existing logic)
            elif self._should_remove_copyright_optimized(stripped_line):
                removal_reason = "copyright_line"
                cleaner_type = "copyright"
            
            # Process result
            if removal_reason:
                # Track stats
                stats_by_type[cleaner_type]["lines_removed"] += 1
                stats_by_type[cleaner_type]["length_reduction"] += len(line) if not self.debug_mode else 0
                stats_by_type[cleaner_type]["removed_lines"].append({
                    "line_content": stripped_line[:100],
                    "line_number": i + 1,
                    "length": len(stripped_line),
                    "reason": removal_reason
                })
                
                if self.debug_mode:
                    # Add debug tag instead of removing
                    debug_tag = f"[DEBUG:{cleaner_type}:{removal_reason}]"
                    cleaned_lines.append(debug_tag)
                else:
                    # Remove line (don't add to cleaned_lines)
                    pass
            else:
                # Keep the line
                cleaned_lines.append(line)
                
            # Update recent lines for repetition detection
            recent_lines.append(stripped_line)
            if len(recent_lines) > 5:  # Max 5 recent lines
                recent_lines.pop(0)
        
        # Combine stats
        combined_stats = {
            "total_lines_removed": sum(s["lines_removed"] for s in stats_by_type.values()),
            "total_length_reduction": sum(s["length_reduction"] for s in stats_by_type.values()),
            "removed_lines": [],
            "doc_id": doc_id or "unknown",
            "cleaning_steps": stats_by_type
        }
        
        # Collect all removed lines (limited)
        for step_stats in stats_by_type.values():
            combined_stats["removed_lines"].extend(step_stats["removed_lines"])
        combined_stats["removed_lines"] = combined_stats["removed_lines"][:10]
        
        return '\n'.join(cleaned_lines), combined_stats
    
    # Helper methods that extract the core logic from each cleaner
    def _should_remove_repetition_optimized(self, line: str, recent_lines: List[str]) -> bool:
        """Extract repetition logic"""
        # Skip debug tags - they can create false positives when different items get same tag
        if line.startswith("[DEBUG:") and line.endswith("]"):
            return False
        
        return line in recent_lines
    
    def _should_remove_tabular_optimized(self, line: str) -> bool:
        """Extract tabular detection logic"""
        should_remove, _ = self.tabular_cleaner._is_tabular_data(line)
        return should_remove
    
    def _should_remove_short_line_optimized(self, line: str) -> bool:
        """Extract short line detection logic"""
        words = line.split()
        word_count = len(words)
        
        # Only process lines with 1-3 words (same as original ShortLineCleaner)
        if word_count <= 3:
            should_remove, _ = self.short_line_cleaner._should_remove_short_line(line, words, word_count)
            return should_remove
        
        return False
    
    def _get_short_line_reason_optimized(self, line: str) -> str:
        """Get specific short line removal reason"""
        words = line.split()
        word_count = len(words)
        
        # Get the actual reason from the original cleaner logic
        should_remove, reason = self.short_line_cleaner._should_remove_short_line(line, words, word_count)
        return reason if reason else "short_line"
    
    def _should_remove_metadata_optimized(self, line: str) -> bool:
        """Extract metadata detection logic"""
        should_remove, _ = self.metadata_cleaner._is_metadata_line(line)
        return should_remove
    
    def _get_metadata_reason_optimized(self, line: str) -> str:
        """Get specific metadata removal reason"""
        should_remove, reason = self.metadata_cleaner._is_metadata_line(line)
        return reason if reason else "metadata_line"
    
    def _should_remove_copyright_optimized(self, line: str) -> bool:
        """Extract copyright detection logic"""
        # Use the compiled patterns from CopyrightCleaner
        return any(pattern.search(line) for pattern in self.copyright_cleaner.compiled_copyright_patterns) 