"""
context validator module

provides generic text analysis methods to evaluate the context of potential citations.
it helps determine if a text segment is an actual citation (to be removed) or a
structural reference, list item, or embedded author-year citation (to be kept).

functions provided:
- `__init__`: initializes the validator with context window and various indicator lists.
- `validate_structural_reference`: validates if a figure/table/section reference should be kept or removed based on context.
- `validate_author_year_citation`: validates author-year citations by assessing their integration into sentence flow.
- `is_embedded_in_sentence_flow`: checks if a text segment is embedded within sentence flow (letters before and after).
- `is_alone_in_line`: determines if a text segment is the sole content on its line.
- `_find_sentence_start`: helper to find the start of the sentence containing a given position.
- `_assess_sentence_integration`: helper to score how well a citation is integrated into a sentence.
- `_is_likely_list_item`: helper to check if a match is likely a list item (e.g., (a), 1.).
"""

import re
from typing import Tuple


class ContextValidator:
    """Validates figure/table/section references based on surrounding context"""
    
    def __init__(self, context_window: int = 150):
        self.context_window = context_window
        
        # Citation indicators - if present, it's a real citation → REMOVE
        self.citation_indicators = [
            "see", "cf.", "compare", "according to", "as shown in", 
            "following", "refer to", "shown in", "presented in", "detailed in"
        ]
        
        # Structural reference indicators - if present, it's structural → KEEP
        self.structural_indicators = [
            "presents", "shows", "reports", "estimates", "demonstrates", 
            "columns", "panel", "specification", "regression", "model",
            "analysis", "results", "findings", "outcome"
        ]
        
        # Flowing text indicators
        self.sentence_indicators = [
            "the", "that", "this", "which", "where", "when", "how", "what",
            "we", "our", "using", "with", "for", "from", "on", "in", "at"
        ]
        
        # Academic verbs that commonly follow author citations (for new method)
        self.academic_verbs = {
            'defined', 'explored', 'investigated', 'studied', 'analyzed', 'examined',
            'found', 'showed', 'demonstrated', 'reported', 'concluded', 'suggested',
            'proposed', 'argued', 'claimed', 'stated', 'noted', 'observed', 
            'discovered', 'revealed', 'indicated', 'pointed', 'emphasized',
            'discussed', 'described', 'explained', 'identified', 'established',
            'conducted', 'performed', 'collected', 'measured', 'compared',
            'developed', 'created', 'designed', 'implemented', 'tested'
        }
    
    def validate_structural_reference(self, match_text: str, start_pos: int, 
                                    end_pos: int, full_text: str) -> Tuple[bool, str]:
        """
        Validate if figure/table/section reference should be kept or removed
        
        Returns:
            (is_valid_to_keep, reason) - True means keep, False means remove
        """
        # Extract context around the match
        context_start = max(0, start_pos - self.context_window)
        context_end = min(len(full_text), end_pos + self.context_window)
        
        before_context = full_text[context_start:start_pos].lower()
        after_context = full_text[end_pos:context_end].lower()
        
        # Check for citation indicators in immediate context (±30 chars)
        immediate_before = before_context[-30:] if len(before_context) >= 30 else before_context
        immediate_after = after_context[:30] if len(after_context) >= 30 else after_context
        
        for indicator in self.citation_indicators:
            if indicator in immediate_before or indicator in immediate_after:
                return False, f"citation_reference_remove ({indicator})"  # REMOVE citations
        
        # Specific check for "in", "in the", "as in" immediately before the match
        immediate_in_patterns = [
            r" in\s*$",       # e.g., "... in Figure 1"
            r" in the\s*$",  # e.g., "... in the Table 2"
            r" as in\s*$"    # e.g., "... as in [3]"
        ]
        for pattern in immediate_in_patterns:
            if re.search(pattern, immediate_before, re.IGNORECASE): # Add IGNORECASE for robustness
                # Extract the matched pattern for logging clarity
                matched_specific_indicator = re.search(pattern, immediate_before, re.IGNORECASE).group(0).strip()
                return False, f"citation_reference_remove (immediate_{matched_specific_indicator.replace(' ', '_')})"
        
        # Check for structural indicators in broader context
        full_context = before_context + after_context
        for indicator in self.structural_indicators:
            if indicator in full_context:
                return True, f"structural_reference_keep ({indicator})"  # KEEP structural
        
        # Count sentence indicators in context (flowing text detection)
        sentence_word_count = sum(1 for word in self.sentence_indicators if word in full_context)
        
        # If many sentence words → likely structural reference in flowing text
        if sentence_word_count >= 3:
            return True, f"flowing_text_keep (sentence_indicators: {sentence_word_count})"  # KEEP flowing text
        
        # Default: if unclear, treat as standalone noise (remove)
        return False, "standalone_noise_remove"  # REMOVE unclear 
    
    def validate_author_year_citation(self, match_text: str, start_pos: int, 
                                    end_pos: int, full_text: str) -> Tuple[bool, str]:
        """
        Validate author-year citations based on sentence integration and flow
        
        Returns:
            (should_remove, reason) - True means remove, False means keep
        """
        # Get context around the match
        context_size = 200  
        before_start = max(0, start_pos - context_size)
        after_end = min(len(full_text), end_pos + context_size)
        
        before_context = full_text[before_start:start_pos]
        after_context = full_text[end_pos:after_end]
        
        # Check if citation is at sentence beginning (author as subject)
        sentence_start = self._find_sentence_start(full_text, start_pos)
        if sentence_start is not None:
            distance_from_start = start_pos - sentence_start
            
            # If citation is very close to sentence start, likely important
            if distance_from_start < 50:  # Within 50 chars of sentence start
                # Check if followed by academic verb
                after_words = after_context.lower().split()[:5]  # First 5 words after citation
                
                for word in after_words:
                    clean_word = re.sub(r'[^\w]', '', word)  # Remove punctuation
                    if clean_word in self.academic_verbs:
                        return False, f"author_as_subject_with_verb_{clean_word}"
                
                # Even without clear verb, sentence-initial citations are often important
                return False, "author_at_sentence_start"
        
        # Check for disruptive reference indicators in context
        before_lower = before_context.lower()
        
        # Patterns that suggest disruptive reference
        if re.search(r'\b(?:see|cf\.|according to|as noted by|as shown by)\s*$', before_lower):
            return True, "disruptive_reference_indicator"
        
        # Check if citation is part of parenthetical reference list
        if re.search(r';\s*$', before_lower) or re.search(r'^\s*;', after_context.lower()):
            return True, "part_of_reference_list"
        
        # Check sentence position and integration
        integration_score = self._assess_sentence_integration(before_context, after_context, match_text)
        
        if integration_score > 0.6:  # Well integrated into sentence flow
            return False, f"well_integrated_score_{integration_score:.2f}"
        elif integration_score < 0.3:  # Poorly integrated, likely just reference
            return True, f"poorly_integrated_score_{integration_score:.2f}"
        else:
            # Uncertain - err on side of keeping for T5 training
            return False, f"uncertain_keep_for_training_score_{integration_score:.2f}"
    
    def _find_sentence_start(self, text: str, position: int) -> int:
        """Find the start of the sentence containing the given position"""
        # Look backwards for sentence boundaries
        search_start = max(0, position - 300)  # Don't search too far back
        
        # Common sentence endings
        sentence_endings = ['. ', '! ', '? ', '\n\n', '\n ']
        
        latest_boundary = search_start
        for ending in sentence_endings:
            pos = text.rfind(ending, search_start, position)
            if pos != -1:
                latest_boundary = max(latest_boundary, pos + len(ending))
        
        return latest_boundary
    
    def _assess_sentence_integration(self, before_context: str, after_context: str, match_text: str) -> float:
        """Assess how well the citation is integrated into sentence flow (0.0 = poor, 1.0 = excellent)"""
        score = 0.5  # Start with neutral score
        
        # Analyze context before citation
        before_words = before_context.lower().split()
        if len(before_words) > 0:
            last_word = before_words[-1]
            
            # Good integration indicators
            if last_word in ['research', 'study', 'work', 'findings', 'analysis']:
                score += 0.2
            elif re.match(r'\w+ing$', last_word):  # Gerund before citation often good
                score += 0.1
                
        # Analyze context after citation  
        after_words = after_context.lower().split()
        if len(after_words) > 0:
            first_word = re.sub(r'[^\w]', '', after_words[0])  # Remove punctuation
            
            # Strong integration: academic verbs after citation
            if first_word in self.academic_verbs:
                score += 0.4
            elif first_word in ['also', 'further', 'additionally', 'moreover']:
                score += 0.2
            elif first_word == 'and':
                score += 0.1
                
        # Check for poor integration indicators
        if re.search(r'\($', before_context.strip()):  # Citation starts parenthetical
            score -= 0.3
        if re.search(r'^\)', after_context.strip()):  # Citation ends parenthetical  
            score -= 0.3
        
        # Ensure score stays in bounds
        return max(0.0, min(1.0, score)) 

    def is_embedded_in_sentence_flow(self, start_pos: int, end_pos: int, full_text: str, context_size: int = 15) -> bool:
        """Check if citation is embedded in sentence flow (letters before AND after)"""
        text_before = full_text[max(0, start_pos - context_size):start_pos]
        text_after = full_text[end_pos:min(len(full_text), end_pos + context_size)]
        
        has_letter_before = bool(re.search(r'[a-zA-Z]', text_before))
        has_letter_after = bool(re.search(r'[a-zA-Z]', text_after))
        
        return has_letter_before and has_letter_after

    def is_alone_in_line(self, start_pos: int, end_pos: int, match_text: str, full_text: str) -> bool:
        """Check if citation is alone in its line"""
        line_start = full_text.rfind('\n', 0, start_pos) + 1
        line_end = full_text.find('\n', end_pos)
        if line_end == -1:
            line_end = len(full_text)
        
        line_content = full_text[line_start:line_end]
        return line_content.strip() == match_text.strip()

    def _is_likely_list_item(self, text_before: str, match_text: str) -> bool:
        """
        Prüft, ob ein Match wahrscheinlich ein Aufzählungspunkt ist.
        Beispiele: (i), (a), 1.
        """
        # Muster für typische Aufzählungen: (a), a), 1., I.
        list_pattern = r"^\s*(\([a-zA-Z0-9]+\)|[a-zA-Z0-9]+\)|[a-zA-Z0-9]+\.|\([ivxlcdm]+\)|[ivxlcdm]+\.)\s*$"
        
        # Prüfen, ob der Match-Text einem typischen Aufzählungsmuster entspricht
        if not re.match(list_pattern, match_text, re.IGNORECASE):
            return False

        # Heuristik 1: Steht am Anfang einer Zeile (oder fast)
        # text_before enthält den Text der Zeile vor dem Match
        if len(text_before.strip()) < 5:  # Toleranz für Leerzeichen oder kleine Einrückungen
            return True

        # Heuristik 2: Folgt auf ein Doppelpunkt am Ende des vorherigen Satzes
        if text_before.strip().endswith(':'):
            return True

        return False 