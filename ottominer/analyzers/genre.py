import json
import logging
from pathlib import Path
from typing import Dict, List, Set
from collections import defaultdict

from .base import AnalyzedDocument

logger = logging.getLogger(__name__)


def _load_json(filename: str) -> dict:
    data_path = Path(__file__).parent.parent / "fdata" / filename
    try:
        with open(data_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Could not load {filename}: {e}")
        return {}


def _flatten(data: dict) -> Set[str]:
    """Recursively flatten a nested dict of lists into a flat set of strings."""
    result = set()
    for v in data.values():
        if isinstance(v, list):
            result.update(s.lower() for s in v)
        elif isinstance(v, dict):
            result.update(_flatten(v))
    return result


_GENRE_DATA = _load_json("genre.json")
_FORMALITY_DATA = _load_json("formality.json")

_FORMAL_MARKERS: Set[str] = {
    m.lower() for m in _FORMALITY_DATA.get("formal_markers", [])
}
_INFORMAL_MARKERS: Set[str] = {
    m.lower() for m in _FORMALITY_DATA.get("informal_markers", [])
}

_PERSIAN_COMPOUNDS: Set[str] = {
    s.lower() for s in _GENRE_DATA.get("persian_compounds", [])
}
_ARABIC_PATTERNS: Set[str] = _flatten(_GENRE_DATA.get("arabic_patterns", {}))
_HONORIFICS: Set[str] = _flatten(_GENRE_DATA.get("honorifics", {}))
_OFFICIAL_DOCS: Set[str] = _flatten(
    {
        "od": _GENRE_DATA.get("legal_and_administrative", {}).get(
            "official_documents", []
        )
    }
)


class GenreAnalyzer:
    """
    Classifies Ottoman documents by register (formal/informal) and genre signals.

    Register is determined by the ratio of formal to informal markers from
    fdata/formality.json. Genre scores count occurrences of Persian compounds,
    Arabic patterns, honorifics, and official document markers from fdata/genre.json.
    """

    def analyze(self, doc: AnalyzedDocument) -> AnalyzedDocument:
        """
        Set doc.register and doc.genre_scores.
        Modifies and returns the same AnalyzedDocument.
        """
        tokens_lower = [t.lower() for t in doc.tokenized.filtered_tokens]

        if not tokens_lower:
            doc.register = "unknown"
            doc.genre_scores = {}
            return doc

        formal_count = sum(1 for t in tokens_lower if t in _FORMAL_MARKERS)
        informal_count = sum(1 for t in tokens_lower if t in _INFORMAL_MARKERS)

        if formal_count == 0 and informal_count == 0:
            doc.register = "unknown"
        elif formal_count > informal_count:
            doc.register = "formal"
        elif informal_count > formal_count:
            doc.register = "informal"
        else:
            doc.register = "mixed"

        total = len(tokens_lower)
        doc.genre_scores = {
            "persian_compound_density": sum(
                1 for t in tokens_lower if t in _PERSIAN_COMPOUNDS
            )
            / total,
            "arabic_pattern_density": sum(
                1 for t in tokens_lower if t in _ARABIC_PATTERNS
            )
            / total,
            "honorific_density": sum(1 for t in tokens_lower if t in _HONORIFICS)
            / total,
            "official_doc_density": sum(1 for t in tokens_lower if t in _OFFICIAL_DOCS)
            / total,
            "formal_marker_count": formal_count,
            "informal_marker_count": informal_count,
        }

        return doc
