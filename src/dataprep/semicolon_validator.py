"""
Semicolon Citation Validator Module

Separate validation logic for semicolon-separated author lists to prevent false positives.
"""

import re
from typing import List, Dict, Any, Tuple, Set

# Stopwords für False Positive Detection
COMMON_STOPWORDS = {
    "the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by",
    "from", "up", "about", "into", "through", "during", "before", "after", "above",
    "below", "between", "among", "this", "that", "these", "those", "a", "an", "as",
    "be", "is", "are", "was", "were", "been", "being", "have", "has", "had", "do",
    "does", "did", "will", "would", "could", "should", "may", "might", "must", "can",
    "form", "but", "however", "therefore", "thus", "also", "only", "just", "even",
    "still", "yet", "already", "now", "then", "here", "there", "where", "when",
    "why", "how", "what", "which", "who", "whom", "whose", "very", "much", "more",
    "most", "many", "some", "any", "each", "every", "all", "both", "either", "neither"
}

# Biologische/lateinische Terminologie - erweitert für OCR Artifacts
BIOLOGICAL_TERMS = {
    "elytris", "prothorace", "punctato", "globosis", "nudis", "sulcato", "fossulato",
    "impunctato", "acutis", "scutello", "hirsutis", "validis", "diviso", "punctulatis",
    "meso", "et", "tibiis", "prothoracis", "protboracis", "corpus", "thorax", "abdomen",
    "species", "genus", "familia", "ordo", "classis", "phylum", "regnum", "metasterni",
    
    # Zusätzliche häufige OCR-Artifacts aus biologischen Texten
    "tarsorum", "antennis", "capite", "pronoto", "mesosterno", "metasterno", 
    "coxis", "femoribus", "segmentis", "marginibus", "angulis", "lateribus",
    "dorsalibus", "ventralibus", "basalibus", "apicalibus", "medianus", "distalis",
    "proximalis", "anterior", "posterior", "superior", "inferior", "external", "internal",
    "specimen", "specimens", "holotype", "paratype", "morphology", "diagnosis", "etymology",
    "material", "examined", "locality", "collection", "museum", "deposited",
    
    # Technische/chemische Begriffe die oft als False Positives erscheinen
    "acetate", "sulfate", "chloride", "hydroxide", "carbonate", "nitrate", "phosphate",
    "oxide", "dioxide", "monoxide", "hydrate", "anhydrous", "crystalline", "amorphous",
    "solvent", "solution", "concentration", "dilution", "extraction", "purification",
    
    # Methodische Begriffe
    "protocol", "procedure", "methodology", "analysis", "measurement", "observation",
    "experiment", "control", "treatment", "sample", "sampling", "preparation"
}


class SemicolonCitationValidator:
    """Simplified validation for semicolon-separated author lists"""
    
    def __init__(
        self,
        max_authors: int = 12,
        min_authors: int = 2,
        confidence_threshold: float = 0.3,  # Much simpler threshold
        context_window: int = 100
    ):
        self.max_authors = max_authors
        self.min_authors = min_authors
        self.confidence_threshold = confidence_threshold
        self.context_window = context_window
    
    def validate(self, match_text: str, start_pos: int, end_pos: int, full_text: str) -> Tuple[bool, Dict[str, Any]]:
        """Simplified validation focusing on stopwords, capitalization, and basic rules"""
        # Extract author candidates
        author_parts = re.split(r'\s*;\s*', match_text.strip())
        author_candidates = [part.strip() for part in author_parts if part.strip()]
        
        validation_info = {
            "author_candidates": author_candidates,
            "num_authors": len(author_candidates),
            "validation_steps": []
        }
        
        # 1. Count check
        if len(author_candidates) < self.min_authors:
            validation_info["validation_steps"].append(f"REJECT: Too few authors ({len(author_candidates)} < {self.min_authors})")
            return False, validation_info
        
        if len(author_candidates) > self.max_authors:
            validation_info["validation_steps"].append(f"REJECT: Too many authors ({len(author_candidates)} > {self.max_authors})")
            return False, validation_info
        
        # 2. Check for obvious stopwords within candidates (more robust)
        for candidate in author_candidates:
            words = candidate.lower().split()
            # Reject if a multi-word candidate contains a stopword, 
            # or if a single-word candidate IS a stopword.
            if any(word in COMMON_STOPWORDS for word in words):
                validation_info["validation_steps"].append(f"REJECT: Candidate '{candidate}' contains/is a stopword")
                return False, validation_info
        
        # 3. Skip biological terms check - too many false negatives
        
        # 4. Check for obvious technical patterns that are NOT author names
        obvious_tech_count = sum(1 for candidate in author_candidates if self._is_obvious_technical_pattern(candidate))
        if obvious_tech_count > 0:
            validation_info["validation_steps"].append(f"REJECT: Contains {obvious_tech_count} obvious technical patterns")
            return False, validation_info
        
        # 5. Skip complex non-author checks - keep it simple
        
        # 6. Simple confidence check - if it passes basic checks, likely a citation
        simple_confidence = self._calculate_simple_confidence(author_candidates)
        validation_info["author_confidence"] = simple_confidence
        
        if simple_confidence >= self.confidence_threshold:
            validation_info["validation_steps"].append(f"ACCEPT: Confidence {simple_confidence:.3f} ≥ {self.confidence_threshold}")
            return True, validation_info
        
        validation_info["validation_steps"].append(f"REJECT: Low confidence ({simple_confidence:.3f} < {self.confidence_threshold})")
        return False, validation_info
    
    def _calculate_simple_confidence(self, author_candidates: List[str]) -> float:
        """Very simple confidence calculation - checks length, and valid characters (capitalization is NOT checked)"""
        if not author_candidates:
            return 0.0
        
        valid_count = 0
        
        for candidate in author_candidates:
            candidate_clean = candidate.strip()
            if not candidate_clean:
                continue
            
            # Simple validity rules:
            # 1. Length between 1-50 chars (more generous for names)
            # 2. Only letters, numbers, basic punctuation, and spaces.
            # 3. Not purely numeric.
            # 4. Not excessively long (e.g. more than 5 words)
            
            words_in_candidate = candidate_clean.split()
            
            is_valid = (
                1 <= len(candidate_clean) <= 50 and
                re.match(r"^[A-Za-z0-9\-'\.\s]+$", candidate_clean) and # Allows spaces
                not candidate_clean.isdigit() and
                len(words_in_candidate) == 1 # MODIFIED: Exactly 1 word per author part
            )
            
            if is_valid:
                valid_count += 1
        
        return valid_count / len(author_candidates) if author_candidates else 0.0
    
    def _is_obvious_technical_pattern(self, candidate: str) -> bool:
        """Erkennt nur die OFFENSICHTLICHSTEN technischen Patterns"""
        candidate_clean = candidate.strip().lower()
        
        # Pattern 1: Nur Zahlen
        if candidate_clean.isdigit():
            return True
        
        # Pattern 2: Enthält mathematische Sonderzeichen
        if re.search(r'[\(\)\[\]{}+=<>|\\/@#$%^&*]', candidate_clean):
            return True
        
        # Pattern 3: Zu lang (über 30 Zeichen)
        if len(candidate_clean) > 30:
            return True
        
        # Pattern 4: Leer oder nur Whitespace
        if not candidate_clean.strip():
            return True
        
        return False
    
    def _looks_like_non_author(self, candidate: str) -> bool:
        """Erkennt offensichtliche Nicht-Autor-Patterns"""
        candidate_clean = candidate.strip().lower()
        
        # Pattern 1: Methodische Begriffe
        method_terms = {
            'method', 'protocol', 'procedure', 'analysis', 'technique', 'approach',
            'algorithm', 'model', 'framework', 'system', 'software', 'program',
            'data', 'result', 'conclusion', 'discussion', 'introduction', 'abstract'
        }
        if candidate_clean in method_terms:
            return True
        
        # Pattern 2: Zeitangaben
        time_patterns = [
            r'^\d{4}$',  # Jahre: 2020, 1995, etc.
            r'^(january|february|march|april|may|june|july|august|september|october|november|december)$',
            r'^(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)$'
        ]
        if any(re.match(pattern, candidate_clean) for pattern in time_patterns):
            return True
        
        # Pattern 3: Institutionen/Orte (häufige False Positives)
        institution_terms = {
            'university', 'institute', 'laboratory', 'department', 'center', 'centre',
            'hospital', 'clinic', 'school', 'college', 'company', 'corporation',
            'foundation', 'society', 'association', 'organization', 'group'
        }
        if any(term in candidate_clean for term in institution_terms):
            return True
        
        # Pattern 4: Geografische Begriffe
        geo_terms = {
            'north', 'south', 'east', 'west', 'central', 'northern', 'southern', 
            'eastern', 'western', 'region', 'area', 'zone', 'district', 'province',
            'state', 'country', 'city', 'town', 'village', 'location'
        }
        if candidate_clean in geo_terms:
            return True
        
        # Pattern 5: Nur Großbuchstaben (oft Abkürzungen, nicht Namen)
        if candidate_clean.isupper() and len(candidate_clean) >= 3:
            return True
        
        # Pattern 6: Enthält Punkt am Ende (oft Abkürzungen)
        if candidate_clean.endswith('.') and len(candidate_clean) <= 5:
            return True
        
        return False 