"""
Figure and Table Line Cleaner Module

Specialized cleaner for removing figure/table references and captions that are
harmful for T5 pretraining, while preserving mathematical content.

Extracted from MultiCitationCleaner to improve modularity and maintainability.
"""

import re
import logging
from typing import Dict, List, Tuple
from src.dataprep.text_cleaners import BaseTextCleaner

log = logging.getLogger(__name__)


class FigureTableCleaner(BaseTextCleaner):
    """
    Cleaner for figure/table references and captions with mathematical content preservation
    """
    
    def __init__(self, debug_mode: bool = False):
        """
        Args:
            debug_mode: If True, add debug tags instead of removing lines
        """
        super().__init__(debug_mode)
        
        # PERFORMANCE OPTIMIZATION: Pre-compile all regex patterns
        self._compile_regex_patterns()
    
    def _compile_regex_patterns(self):
        """Pre-compile all regex patterns for optimal performance"""
        # Figure/table caption pattern
        self.compiled_figure_pattern = re.compile(
            r'^(?:fig|figure|tab|table|tbl)\.?\s*\d+(?:\.\d+)?[\s:]*', 
            re.IGNORECASE
        )
        
        # Numeric word pattern
        self.compiled_numeric_pattern = re.compile(r'^\d+\.?\d*$')
        
        # Mathematical content patterns
        math_patterns = [
            r'[αβγδεζηθικλμνξοπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ]',  # Greek letters
            r'[∂∇∑∏∫≡≠≤≥±∓×÷∈∉⊂⊃∪∩]',  # Math operators
            r'⟨[^⟩]*⟩',                    # Bra-ket notation
            r'\|[^|]*⟩',                   # Ket notation  
            r'⟨[^|]*\|',                   # Bra notation
            r'[a-zA-Z]+_[a-zA-Z0-9]+',     # Subscripts like γA, γB
            r'[a-zA-Z]+\^[a-zA-Z0-9]+',    # Superscripts
            r'\\[a-z]+\{[^}]*\}',          # LaTeX commands
            r'[a-zA-Z]+=.*[a-zA-Z]',       # Equations like tl=t0[...]
            r'ℋ|ℰ|ℱ|ℊ|ℋ|ℌ|ℍ|ℎ|ℏ|ℐ',      # Script letters (Hamiltonian etc.)
            r'Π†|Π|†|‡|°|µ|σ|ρ|τ|φ|χ|ψ|ω', # Special math symbols
        ]
        
        self.compiled_math_patterns = [re.compile(pattern) for pattern in math_patterns]
        
        log.info(f"Compiled figure/table patterns: {len(self.compiled_math_patterns)} math patterns + 2 other patterns")
    
    def clean(self, text: str, doc_id: str = None) -> Tuple[str, Dict]:
        """
        Remove lines containing figure/table references and captions aggressively with detailed logging
        
        Args:
            text: Input text to clean
            doc_id: Document ID for logging (optional)
            
        Returns:
            Tuple of (cleaned_text, stats_dict)
        """
        lines = text.splitlines()
        cleaned_lines = []
        removed_lines = []
        
        for i, line in enumerate(lines):
            stripped_line = line.strip()
            
            # Skip empty lines
            if not stripped_line:
                cleaned_lines.append(line)
                continue
            
            # OPTIMIZED Figure/Table line detection - specialized patterns only
            should_remove = False
            removal_reason = ""
            
            # Pattern 1: Clear figure/table captions (e.g., "Figure 2.1:", "Table 5")
            if self.compiled_figure_pattern.match(stripped_line):
                should_remove = True
                removal_reason = "figure_table_caption"
            
            elif not self._is_mathematical_content(stripped_line):  
                words = stripped_line.split()
                if 2 <= len(words) <= 5:
                    numeric_words = sum(1 for word in words if self.compiled_numeric_pattern.match(word))
                    if numeric_words / len(words) >= 0.6:
                        should_remove = True
                        removal_reason = f"numeric_table_data_{numeric_words}of{len(words)}"
            
            if should_remove:
                # Store removed line for analytics
                removed_lines.append({
                    "line_content": stripped_line[:100],  # Limit length for W&B
                    "line_number": i + 1,
                    "length": len(stripped_line),
                    "reason": removal_reason
                })
                
                if self.debug_mode:
                    # Debug-Tag statt Entfernung
                    debug_tag = f"[DEBUG:figure_line:{removal_reason}]"
                    cleaned_lines.append(debug_tag)
                else:
                    # Line wird nicht hinzugefügt (entfernt)
                    continue
            else:
                # Keep the line
                cleaned_lines.append(line)
        
        # Statistics
        stats = {
            "lines_removed": len(removed_lines),
            "length_reduction": sum(item["length"] for item in removed_lines) if not self.debug_mode else 0,
            "removed_lines": removed_lines[:10],  # Store max 10 samples per document
            "doc_id": doc_id or "unknown"
        }
        
        return '\n'.join(cleaned_lines), stats
    
    def _is_mathematical_content(self, line: str) -> bool:
        """Check if line contains mathematical formulas that should be preserved"""
        # Mathematical patterns that indicate scientific content
        math_patterns = [
            r'[αβγδεζηθικλμνξοπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ]',  # Greek letters
            r'[∂∇∑∏∫≡≠≤≥±∓×÷∈∉⊂⊃∪∩]',  # Math operators
            r'⟨[^⟩]*⟩',                    # Bra-ket notation
            r'\|[^|]*⟩',                   # Ket notation  
            r'⟨[^|]*\|',                   # Bra notation
            r'[a-zA-Z]+_[a-zA-Z0-9]+',     # Subscripts like γA, γB
            r'[a-zA-Z]+\^[a-zA-Z0-9]+',    # Superscripts
            r'\\[a-z]+\{[^}]*\}',          # LaTeX commands
            r'[a-zA-Z]+=.*[a-zA-Z]',       # Equations like tl=t0[...]
            r'ℋ|ℰ|ℱ|ℊ|ℋ|ℌ|ℍ|ℎ|ℏ|ℐ',      # Script letters (Hamiltonian etc.)
            r'Π†|Π|†|‡|°|µ|σ|ρ|τ|φ|χ|ψ|ω', # Special math symbols
        ]
        
        return any(pattern.search(line) for pattern in self.compiled_math_patterns) 