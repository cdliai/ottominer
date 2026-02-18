import json
import logging
from pathlib import Path
from typing import Dict, List
from collections import defaultdict

from .base import AnalyzedDocument

logger = logging.getLogger(__name__)


def _load_suffix_table() -> Dict[str, List[str]]:
    """
    Load suffixes.json and return a flat map: suffix_string -> [suffix_type, ...].
    suffixes.json structure: {"case_markers": [...], "verbal": [...], ...}
    """
    data_path = Path(__file__).parent.parent / "fdata" / "suffixes.json"
    table: Dict[str, List[str]] = defaultdict(list)
    try:
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for suffix_type, suffixes in data.items():
            for suffix in suffixes:
                table[suffix].append(suffix_type)
    except Exception as e:
        logger.error(f"Could not load suffixes.json: {e}")
    return dict(table)


_SUFFIX_TABLE: Dict[str, List[str]] = _load_suffix_table()
_SORTED_SUFFIXES: List[str] = sorted(_SUFFIX_TABLE.keys(), key=len, reverse=True)


class MorphologyAnalyzer:
    """
    Rule-based suffix analyzer for Ottoman Turkish.

    Uses fdata/suffixes.json (case markers, possessive, plural, verbal,
    derivational, diminutive, causative, agentive). Applies longest-match
    suffix stripping. One token can match multiple suffix types.
    """

    def __init__(self):
        self._table = _SUFFIX_TABLE
        self._sorted = _SORTED_SUFFIXES

    def analyze_token(self, token: str) -> List[str]:
        """
        Return a list of suffix type names matched on the token (longest match first).
        Returns empty list if no suffix matches.
        """
        token_lower = token.lower()
        matched_types: List[str] = []
        seen_types = set()

        for suffix in self._sorted:
            if token_lower.endswith(suffix) and len(token_lower) > len(suffix):
                for suffix_type in self._table[suffix]:
                    if suffix_type not in seen_types:
                        matched_types.append(suffix_type)
                        seen_types.add(suffix_type)

        return matched_types

    def analyze(self, doc: AnalyzedDocument) -> AnalyzedDocument:
        """
        Annotate doc.morphology for each token in filtered_tokens.
        Only tokens with at least one suffix match are stored.
        Modifies and returns the same AnalyzedDocument.
        """
        morphology: Dict[str, List[str]] = {}
        for token in doc.tokenized.filtered_tokens:
            result = self.analyze_token(token)
            if result:
                morphology[token] = result
        doc.morphology = morphology
        return doc

    def aggregate_stats(self, doc: AnalyzedDocument) -> Dict:
        """
        Return suffix type frequency counts across all analyzed tokens.
        """
        suffix_type_counts: Dict[str, int] = defaultdict(int)
        for types in doc.morphology.values():
            for t in types:
                suffix_type_counts[t] += 1

        total = len(doc.tokenized.filtered_tokens)
        annotated = len(doc.morphology)

        return {
            "suffix_type_counts": dict(suffix_type_counts),
            "annotated_token_count": annotated,
            "total_token_count": total,
            "annotation_rate": annotated / total if total > 0 else 0.0,
        }
