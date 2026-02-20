"""
Symbol Token Normalizer for T5

Normalizes scientific symbols for better T5 tokenization.
"""

import re
import logging
from typing import Dict, Tuple
from datatrove.data import Document
from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.pipeline.writers.disk_base import DiskWriter

log = logging.getLogger(__name__)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class SymbolTokenNormalizer(BaseFilter):
    
    name = "Symbol Token Normalizer"
    
    def __init__(
        self,
        debug_mode: bool = False,
        log_to_wandb: bool = True,
        exclusion_writer: DiskWriter = None
    ):
        super().__init__(exclusion_writer)
        self.debug_mode = debug_mode
        self.log_to_wandb = log_to_wandb and WANDB_AVAILABLE
        
        # Math operators - ULTRA-CONSERVATIVE: minimal normalization only
        self.math_operators = {
            # REMOVED: ≤≥≠± (preserve mathematical Unicode - visual clarity, standard in scientific texts)
            # REMOVED: ×÷ (cross product vs scalar multiplication), ≈≅≃∝≡≢ (preserve mathematical meaning)
            # REMOVED: ⋅· (dot product vs multiplication - semantically different)
            # REMOVED: ° (degree symbol - preserve for scientific units like 20°C, 45°)
            r'′': "'", r'″': "''",  # Only prime symbols (accessible via keyboard)
            # PRESERVED: All mathematical Unicode (≤≥≠±×÷°∂∇∆∞) and Greek letters (α,β,γ) for scientific authenticity
        }
        
        # Math symbols - CONSERVATIVE: only normalize clearly problematic symbols
        self.math_symbols = {
            # DISABLED: Keep ∑, ∏, ∫, √ as Unicode for scientific authenticity
            # r'∑': ' sum ', r'∏': ' prod ', r'∫': ' integral ', r'∮': ' contour_integral ',
            # r'√': ' sqrt ', r'∛': ' cbrt ', r'∜': ' sqrt4 ',
            # DISABLED: Keep ceiling/floor as Unicode
            # r'⌈': ' ceil ', r'⌉': ' ceil ', r'⌊': ' floor ', r'⌋': ' floor ',
            # r'|': 'absolute',  # REMOVED - keep | symbol as-is for absolute values |x|
            # DISABLED: Keep parallel/perp as Unicode  
            # r'∥': ' parallel ', r'⊥': ' perp ',
            r'…': '...', r'⋯': '...', # Keep ellipsis normalization (improves readability)
            # DISABLED: Keep mathematical dots as Unicode
            # r'⋮': ' vdots ', r'⋱': ' ddots ',
        }
        
        # Logic and set symbols - DISABLED to preserve scientific notation
        self.logic_set_symbols = {
            # DISABLED: Keep ∧, ∨, ¬, ∀, ∃, ∈, ⊂, ∪, ∩, etc. as Unicode
            # Scientific texts benefit from preserving these symbols
        }
        
        # Fractions - DISABLED: preserve Unicode for scientific authenticity
        self.fractions = {
            # DISABLED: Keep ½⅓¼¾ as Unicode - T5 can handle these well
            # Scientific texts benefit from visual clarity of Unicode fractions
        }
        
        # Superscripts - DISABLED: preserve Unicode for scientific authenticity
        self.superscripts = {
            # DISABLED: Keep ¹²³⁴⁵ as Unicode - iconic formulas like E=mc², x², H₂O
            # T5 tokenizer handles Unicode superscripts well, visual clarity important
        }
        
        # Subscripts - DISABLED: preserve Unicode for scientific authenticity
        self.subscripts = {
            # DISABLED: Keep ₁₂₃₄₅ as Unicode - chemical formulas like H₂O, CO₂
            # Scientific notation benefits from Unicode subscripts
        }
        
        # Quotation marks and guillemets
        self.quotation_marks = {
            r'"': '"', r'"': '"', r''': "'", r''': "'",
            r'‚': ',', r'„': '"', r'‹': '<', r'›': '>',
            r'«': '<<', r'»': '>>', r'‹': '<', r'›': '>',
            r'❛': "'", r'❜': "'", r'❝': '"', r'❞': '"'
        }
        
        # Legal symbols (minimal - only common ones in scientific texts)
        self.legal_currency = {
            r'©': '(c)', r'®': '(R)', r'™': '(TM)',
            r'§': 'section', r'¶': 'paragraph',
            # CURRENCY SYMBOLS NOT HANDLED: €, £, ¥, $, etc. stay as-is (not replaced, not removed)
        }
        
        # Greek letters - DISABLED to preserve Unicode for scientific texts
        # T5 with SentencePiece handles Unicode well, and scientific meaning is preserved
        self.greek_letters = {
            # Disabled: keep α, β, γ, etc. as Unicode for scientific authenticity
        }
        
        # Scientific notation - DISABLED: preserve original for scientific texts
        self.scientific_notation = {
            # DISABLED: Keep original notation (1.23e-4) - T5 tokenizer handles this well
            # Normalization would lose information and reduce precision
        }
        
        # Bullet points - DEDUPLICATED
        self.bullet_points = {
            r'•': '*', r'‣': '*', r'‧': '*', r'⁃': '*',
            r'◦': '*', r'⁌': '*', r'⁍': '*'
        }
        
        
        # Arrows - CONSERVATIVE: only common ASCII-equivalent arrows
        self.arrows = {
            r'→': '->', r'←': '<-', r'↔': '<->',  # Common directional arrows
            r'⇒': '=>', r'⇐': '<=', r'⇔': '<=>',  # Logic arrows (implication)
            r'⟶': '-->', r'⟵': '<--', r'⟷': '<-->'  # Long arrows
            # REMOVED: Keep ↑↓↗↘↙↖ as Unicode - no clear ASCII standard, and they're visually clear
        }
        
        # Special minus and dashes
        self.special_minus = {
            r'−': '-', r'‐': '-', r'–': '-', r'—': '-'
        }
        
        # Invisible mathematical characters - FIXED regex syntax
        self.invisible_chars = {
            r'\u2062': '', r'\u2063': '', r'\u2061': ''  # Function application, separator, invisible times
        }
        
        # Cyrillic to Latin (common OCR errors)
        self.cyrillic_to_latin = {
            r'а': 'a', r'е': 'e', r'о': 'o', r'р': 'p', r'с': 'c',
            r'у': 'y', r'х': 'x', r'А': 'A', r'В': 'B', r'Е': 'E',
            r'К': 'K', r'М': 'M', r'Н': 'H', r'О': 'O', r'Р': 'P',
            r'С': 'C', r'Т': 'T', r'У': 'Y', r'Х': 'X', r'и': 'u',
            r'н': 'h', r'т': 't', r'ь': 'b'
        }
        
        # Table of Contents and structural patterns (REMOVE completely)
        self.toc_patterns = {
            r'\w+[.\s]{5,}\d+': '',                    # "Introduction ... 5" -> remove
            r'^[.\t\s]{10,}$': '',                     # Lines of only dots/tabs -> remove
            r'\w+[.\t]{5,}\w+': '',                    # "Chapter\t\t\tTitle" -> remove
        }
        
        # Tabular data patterns (REMOVE completely)  
        self.tabular_patterns = {
            r'^[\d\t\s\.]{20,}$': '',                  # Lines mostly numbers/tabs -> remove
            r'^.*\t.*\t.*\t.*$': '',                   # 4+ tab-separated values -> remove
        }
        
        # Excessive dots (SHORTEN to ellipsis)
        self.dot_shortening = {
            r'\.{4,}': '...',                          # 4+ dots -> ellipsis
            r'(\. ){3,}': ' ... ',                     # Spaced dots -> ellipsis
        }
        
        self.custom_stats = {
            "docs_processed": 0,
            "docs_normalized": 0,
            "total_normalizations": 0
        }
        
        self.wandb_initialized = False
        self.current_rank = 0
        
        # PERFORMANCE OPTIMIZATION: Pre-compile all regex patterns
        self._compile_all_patterns()
    
    def _compile_all_patterns(self):
        """Pre-compile all regex patterns for optimal performance"""
        # Collect all patterns in processing order
        pattern_groups = [
            self.invisible_chars,  # Remove invisible chars first
            self.toc_patterns,  # Remove table of contents
            self.tabular_patterns,  # Remove tabular data
            self.dot_shortening,  # Shorten excessive dots
            self.cyrillic_to_latin,  # Fix OCR errors
            self.special_minus,  # Normalize dashes
            self.quotation_marks,  # Normalize quotes
            self.bullet_points,  # Normalize bullets
            self.arrows,  # Normalize arrows
            self.fractions,  # Normalize fractions
            self.superscripts,  # Normalize superscripts
            self.legal_currency,  # Normalize legal/currency symbols
            self.logic_set_symbols,  # Logic and set symbols
            self.math_operators,  # Math operators
            self.math_symbols,  # Math symbols
            self.greek_letters,  # Greek letters
            self.scientific_notation  # Scientific notation
        ]
        
        # Pre-compile all patterns for performance
        self.compiled_patterns = []
        for patterns in pattern_groups:
            for pattern, replacement in patterns.items():
                try:
                    compiled_pattern = re.compile(pattern)
                    self.compiled_patterns.append((compiled_pattern, replacement))
                except re.error as e:
                    log.warning(f"Invalid regex pattern '{pattern}': {e}")
                    continue
        
        # For debug mode, compile all patterns into one group
        if self.debug_mode:
            debug_patterns = {**self.math_operators, **self.math_symbols, **self.logic_set_symbols,
                           **self.fractions, **self.superscripts, **self.quotation_marks,
                           **self.legal_currency, **self.greek_letters, **self.bullet_points, **self.arrows}
            self.debug_compiled_patterns = []
            for pattern, replacement in debug_patterns.items():
                try:
                    compiled_pattern = re.compile(pattern)
                    self.debug_compiled_patterns.append((compiled_pattern, replacement))
                except re.error as e:
                    log.warning(f"Invalid debug regex pattern '{pattern}': {e}")
                    continue
        
        log.info(f"Compiled {len(self.compiled_patterns)} regex patterns for symbol normalization")
    
    def _normalize_symbols(self, text: str) -> Tuple[str, int]:
        normalized_text = text
        normalizations = 0
        
        if self.debug_mode:
            # Use pre-compiled debug patterns
            for compiled_pattern, replacement in self.debug_compiled_patterns:
                matches = compiled_pattern.findall(normalized_text)
                if matches:
                    normalized_text = compiled_pattern.sub(f'[DEBUG:symbol:{replacement}]', normalized_text)
                    normalizations += len(matches)
            return normalized_text, normalizations
        
        # Use pre-compiled patterns for optimal performance
        for compiled_pattern, replacement in self.compiled_patterns:
            matches = compiled_pattern.findall(normalized_text)
            if matches:
                normalized_text = compiled_pattern.sub(replacement, normalized_text)
                normalizations += len(matches)
        
        return normalized_text, normalizations
    
    def filter(self, doc: Document) -> bool:
        original_text = doc.text
        self.custom_stats["docs_processed"] += 1
        
        normalized_text, normalizations_count = self._normalize_symbols(original_text)
        doc.text = normalized_text
        
        text_changed = original_text != normalized_text
        if text_changed:
            self.custom_stats["docs_normalized"] += 1
            self.custom_stats["total_normalizations"] += normalizations_count
            
            doc.metadata["symbol_text_changed"] = True
            doc.metadata["symbol_normalizations"] = normalizations_count
        else:
            doc.metadata["symbol_text_changed"] = False
            doc.metadata["symbol_normalizations"] = 0
        
        self.stat_update("docs_processed")
        if text_changed:
            self.stat_update("docs_with_symbol_normalizations")
            self.stat_update("total_symbol_normalizations", normalizations_count)
        
        if self.current_rank == 0 and self.wandb_initialized and self.custom_stats["docs_processed"] % 100 == 0:
            self._log_stats()
        
        return bool(normalized_text.strip())
    
    def _log_stats(self):
        if not self.wandb_initialized:
            return
            
        try:
            wandb.log({
                "symbol_norm/docs_processed": self.custom_stats["docs_processed"],
                "symbol_norm/docs_normalized": self.custom_stats["docs_normalized"],
                "symbol_norm/normalization_rate": (
                    self.custom_stats["docs_normalized"] / self.custom_stats["docs_processed"] 
                    if self.custom_stats["docs_processed"] > 0 else 0
                ),
                "symbol_norm/total_normalizations": self.custom_stats["total_normalizations"]
            })
        except Exception as e:
            log.warning(f"Failed to log symbol normalization stats: {e}")
    
    def run(self, data, rank: int = 0, world_size: int = 1):
        self.current_rank = rank
        
        if rank == 0 and self.log_to_wandb:
            self._init_wandb()
        
        try:
            yield from super().run(data, rank, world_size)
        finally:
            log.info(f"Symbol normalization rank {rank}: {self.custom_stats['docs_processed']} docs, "
                    f"{self.custom_stats['docs_normalized']} normalized, "
                    f"{self.custom_stats['total_normalizations']} symbols")
    
    def _init_wandb(self):
        try:
            if wandb.run is not None:
                self.wandb_initialized = True
                log.info("Using existing W&B session for Symbol Normalizer")
            else:
                wandb.init(
                    project="BA-DataTrove",
                    group="symbol-normalization",
                    tags=["symbol-normalization", "t5-optimization"],
                    job_type="symbol-normalization",
                    config={"debug_mode": self.debug_mode}
                )
                self.wandb_initialized = True
                log.info("W&B initialized for Symbol Normalizer")
        except Exception as e:
            log.warning(f"Failed to initialize W&B: {e}")
            self.log_to_wandb = False