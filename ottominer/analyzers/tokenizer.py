import re
import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Set

from .base import TokenizedDocument

logger = logging.getLogger(__name__)

# Try to import durak; fall back to regex if unavailable.
# The actual durak-nlp API returns plain strings from tokenize(), not
# token objects with .text/.offset attributes, and does not expose Lemmatizer.
try:
    from durak import clean_text as durak_clean, tokenize as durak_tokenize
    _DURAK_AVAILABLE = True
    logger.info("durak-nlp loaded successfully")
except ImportError:
    _DURAK_AVAILABLE = False
    logger.warning(
        "durak-nlp not installed. Falling back to regex tokenizer. "
        "Install with: pip install durak-nlp"
    )


def _load_ottoman_stopwords() -> Set[str]:
    """Load the 174-term Ottoman stopword list from fdata/stopwords.json."""
    data_path = Path(__file__).parent.parent / "fdata" / "stopwords.json"
    try:
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return set(data.get("particles_and_conjunctions", []))
    except Exception as e:
        logger.error(f"Could not load Ottoman stopwords: {e}")
        return set()


_OTTOMAN_STOPWORDS: Set[str] = _load_ottoman_stopwords()

# Regex tokenizer used when durak is unavailable
_TOKEN_RE = re.compile(
    r"[a-zA-ZğĞıİöÖşŞüÜçÇ\u0600-\u06FF]+",
    re.UNICODE,
)


def _regex_tokenize(text: str) -> Tuple[List[str], List[Tuple[int, int]]]:
    """Simple regex fallback tokenizer. Returns (tokens, offsets)."""
    tokens = []
    offsets = []
    for match in _TOKEN_RE.finditer(text):
        tokens.append(match.group())
        offsets.append((match.start(), match.end()))
    return tokens, offsets


def _compute_offsets(text: str, tokens: List[str]) -> List[Tuple[int, int]]:
    """
    Given a text and a list of tokens (strings), find each token's character
    span in `text` using a sequential scan so offsets never go backwards.
    Works correctly even when durak lowercases the text.
    """
    offsets = []
    cursor = 0
    text_lower = text.lower()
    for tok in tokens:
        tok_lower = tok.lower()
        idx = text_lower.find(tok_lower, cursor)
        if idx == -1:
            # Token not found from cursor — search from beginning as fallback
            idx = text_lower.find(tok_lower)
        if idx == -1:
            # Still not found; use current cursor as best-effort placeholder
            idx = cursor
        end = idx + len(tok)
        offsets.append((idx, end))
        cursor = end
    return offsets


def _durak_pipeline(text: str) -> Tuple[List[str], List[str], List[Tuple[int, int]]]:
    """
    Run the durak pipeline.

    Returns (tokens, lemmas, offsets).

    Notes on the actual durak API (differs from original plan):
    - durak.tokenize() returns List[str] (plain strings), not token objects.
    - durak.clean_text() lowercases and normalises Unicode but preserves Turkish
      characters (ş, ğ, ç, etc.).
    - durak.Lemmatizer does not exist in this version; surface forms are used.
    """
    cleaned = durak_clean(text)
    tokens: List[str] = durak_tokenize(cleaned)
    # Compute character offsets against the cleaned (lowercased) text so that
    # the sequential scan works correctly.
    offsets = _compute_offsets(cleaned, tokens)
    # No Lemmatizer available in this durak version — use surface forms.
    lemmas = tokens[:]
    return tokens, lemmas, offsets


class OttomanTokenizer:
    """
    Tokenizer for Ottoman Turkish text.

    Uses durak-nlp for tokenization mechanics (Unicode normalization,
    suffix attachment, best-effort lemmatization) and applies Ottoman-specific
    stopword filtering from fdata/stopwords.json on top.

    When durak is unavailable, falls back to regex tokenization with a warning.
    Ottoman stopword filtering is always applied regardless of backend.
    """

    def __init__(self, extra_stopwords: Optional[Set[str]] = None):
        self.stopwords = set(_OTTOMAN_STOPWORDS)
        if extra_stopwords:
            self.stopwords.update(extra_stopwords)

    def tokenize(self, source_path: str, text: str) -> TokenizedDocument:
        """
        Tokenize text and return a TokenizedDocument.

        Args:
            source_path: Original file path (stored in document, not read here).
            text: Raw text to tokenize.

        Returns:
            TokenizedDocument with tokens, lemmas, offsets, and filtered tokens.
        """
        if not text or not text.strip():
            return TokenizedDocument(
                source_path=source_path,
                raw_text=text,
                tokens=[],
                lemmas=[],
                offsets=[],
                filtered_tokens=[],
            )

        if _DURAK_AVAILABLE:
            try:
                tokens, lemmas, offsets = _durak_pipeline(text)
            except Exception as e:
                logger.warning(f"durak pipeline failed ({e}), falling back to regex")
                tokens, offsets = _regex_tokenize(text)
                lemmas = tokens[:]
        else:
            tokens, offsets = _regex_tokenize(text)
            lemmas = tokens[:]

        filtered = [t for t in tokens if t.lower() not in self.stopwords]

        return TokenizedDocument(
            source_path=source_path,
            raw_text=text,
            tokens=tokens,
            lemmas=lemmas,
            offsets=offsets,
            filtered_tokens=filtered,
        )
