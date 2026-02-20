import logging
from typing import Dict, List, Optional
from .base import AnalyzedDocument
from .morphology import MorphologyAnalyzer

logger = logging.getLogger(__name__)

try:
    import durak
    _DURAK_AVAILABLE = True
except ImportError:
    _DURAK_AVAILABLE = False


class DurakAnalyzer:
    """
    Durak-backed analyzer for Turkish morphology.
    
    Acts as a high-speed wrapper around durak-nlp for Latin-script Ottoman.
    Currently, it normalizes detached suffixes and falls back to the rule-based
    MorphologyAnalyzer for historical Ottoman suffixes. As durak expands its
    Arabic script and historical Turkish support, this module will replace the
    rule-based fallback entirely.
    """

    def __init__(self):
        self.fallback_morphology = MorphologyAnalyzer()
        if not _DURAK_AVAILABLE:
            logger.warning("durak-nlp is not installed. DurakAnalyzer will rely entirely on fallback.")

    def analyze(self, doc: AnalyzedDocument) -> AnalyzedDocument:
        """
        Perform morphological analysis using durak where possible,
        and rule-based fallback for Ottoman-specific vocabulary.
        """
        # If durak is not available, delegate completely to fallback
        if not _DURAK_AVAILABLE:
            return self.fallback_morphology.analyze(doc)
            
        morphology: Dict[str, List[str]] = {}
        
        # Step 1: Use durak to handle detached suffixes and clean tokens if necessary.
        # Currently, durak's attach_detached_suffixes operates on a list of tokens.
        tokens = doc.tokenized.filtered_tokens
        try:
            # Try to attach detached suffixes using durak (e.g. "yapacak mı" -> "yapacakmı")
            # In a full durak pipeline, this would happen during tokenization,
            # but we can apply it here as part of structural analysis.
            normalized_tokens = durak.attach_detached_suffixes(tokens)
        except Exception as e:
            logger.debug(f"durak.attach_detached_suffixes failed: {e}. Using original tokens.")
            normalized_tokens = tokens

        # Step 2: Lemmatization / Morphology
        # As durak-nlp currently does not expose a full Lemmatizer in this version,
        # we will use the rule-based suffix fallback to populate the morphology dictionary.
        # In the future, this loop will be replaced by:
        # for token in normalized_tokens:
        #     morphology[token] = durak.morphological_analysis(token)
        
        seen_tokens = set()
        for i, token in enumerate(normalized_tokens):
            if token in seen_tokens:
                continue
                
            seen_tokens.add(token)
            
            # Since durak doesn't have an Arabic/Ottoman morphology engine yet,
            # use our fallback rule-based analyzer for the Ottoman specifics.
            suffix_types = self.fallback_morphology.analyze_token(token)
            
            if suffix_types:
                morphology[token] = suffix_types

        # Update the document's morphology and filtered tokens (in case durak merged some)
        doc.morphology = morphology
        # If durak merged detached suffixes, we might want to update filtered_tokens,
        # but changing filtered_tokens might break offsets mapping. 
        # For now, just enrich the morphology dict.
        
        return doc
