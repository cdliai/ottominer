import json
import logging
from pathlib import Path
from typing import Dict, List
from collections import defaultdict

from .base import AnalyzedDocument

logger = logging.getLogger(__name__)


def _load_semantic_index() -> Dict[str, List[str]]:
    """
    Build a reverse index: token -> [category, ...] from semantics.json.
    semantics.json structure: {"religious": [...], "economic": [...], ...}
    """
    data_path = Path(__file__).parent.parent / "fdata" / "semantics.json"
    index: Dict[str, List[str]] = defaultdict(list)
    try:
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for category, terms in data.items():
            for term in terms:
                index[term.lower()].append(category)
    except Exception as e:
        logger.error(f"Could not load semantics.json: {e}")
    return dict(index)


_SEMANTIC_INDEX: Dict[str, List[str]] = _load_semantic_index()


class SemanticAnalyzer:
    """
    Labels tokens with Ottoman semantic categories.

    Categories come from fdata/semantics.json:
    religious, cultural, emotional, political, economic,
    educational, legal, social, intellectual, geographical, scientific.

    One token can carry multiple labels.
    """

    def __init__(self):
        self._index = _SEMANTIC_INDEX

    def label_token(self, token: str) -> List[str]:
        """Return a list of semantic categories for a token (empty if unknown)."""
        return self._index.get(token.lower(), [])

    def analyze(self, doc: AnalyzedDocument) -> AnalyzedDocument:
        """
        Annotate doc.semantic_labels for every token in filtered_tokens.
        Modifies and returns the same AnalyzedDocument.
        """
        labels: Dict[str, List[str]] = {}
        for token in doc.tokenized.filtered_tokens:
            result = self.label_token(token)
            if result:
                labels[token] = result
        doc.semantic_labels = labels
        return doc

    def aggregate_stats(self, doc: AnalyzedDocument) -> Dict:
        """
        Return aggregate statistics over semantic_labels:
        - category_counts: how many tokens per category
        - dominant_category: category with most tokens
        - coverage: fraction of filtered_tokens with at least one label
        """
        category_counts: Dict[str, int] = defaultdict(int)
        for cats in doc.semantic_labels.values():
            for cat in cats:
                category_counts[cat] += 1

        total = len(doc.tokenized.filtered_tokens)
        labeled = len(doc.semantic_labels)
        coverage = labeled / total if total > 0 else 0.0
        dominant = max(category_counts, key=category_counts.get) if category_counts else None

        return {
            "category_counts": dict(category_counts),
            "dominant_category": dominant,
            "coverage": coverage,
            "labeled_token_count": labeled,
            "total_token_count": total,
        }
