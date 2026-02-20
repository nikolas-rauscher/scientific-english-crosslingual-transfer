"""Document-level language cleaner for T5 pretraining using FastText scores"""

import logging
from typing import Dict, Tuple
from src.dataprep.text_cleaners import BaseTextCleaner

log = logging.getLogger(__name__)


class SmartLanguageCleaner(BaseTextCleaner):
    """Document-level language cleaner using precomputed fasttext_en scores"""
    
    def __init__(self, fasttext_threshold: float = 0.75, debug_mode: bool = False):
        super().__init__(debug_mode)
        self.fasttext_threshold = fasttext_threshold
        log.info(f"SmartLanguageCleaner initialized with threshold={fasttext_threshold}")
    
    def clean(self, text: str, doc_id: str = None, fasttext_en: float = 1.0) -> Tuple[str, Dict]:
        """Clean text based on document-level FastText English confidence score.
        
        Args:
            text: Input text to clean
            doc_id: Document identifier for logging
            fasttext_en: FastText English confidence score (0.0-1.0)
            
        Returns:
            Tuple of (cleaned_text, stats_dict)
            - If fasttext_en >= threshold: returns original text
            - If fasttext_en < threshold: returns empty string (document removed)
        """
        original_length = len(text)
        
        if fasttext_en >= self.fasttext_threshold:
            return text, {
                "document_kept": True,
                "reason": f"good_fasttext_en_score_{fasttext_en:.3f}",
                "fasttext_score": fasttext_en,
                "original_length": original_length,
                "length_reduction": 0,
                "doc_id": doc_id or "unknown"
            }
        else:
            # Remove entire document if FastText score is too low
            return "", {
                "document_removed": True,
                "reason": f"low_fasttext_en_score_{fasttext_en:.3f}",
                "fasttext_score": fasttext_en,
                "original_length": original_length,
                "length_reduction": original_length,
                "doc_id": doc_id or "unknown"
            }