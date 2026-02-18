# OttoMiner Overhaul Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Transform OttoMiner from a ~60% complete PDF extraction tool into a fully working Ottoman Turkish NLP pipeline with tiered extraction, durak-backed tokenization, Ottoman-specific analyzers, and HTML + matplotlib visualization.

**Architecture:** A `Pipeline` class chains five stages — Extract → Tokenize → Analyze → Output → Visualize. Each stage is independently testable. The `analyzers/` layer wraps `durak` for tokenization mechanics and applies Ottoman-specific JSON lexicons on top. The `visualizers/` layer is a pure consumer of pipeline output JSON.

**Tech Stack:** pymupdf4llm (Tier 1 extraction), chardet (encoding repair), surya (Tier 2 OCR, optional), durak-nlp (tokenization), scikit-learn (TF-IDF similarity), plotly (HTML reports), matplotlib (static figures), rich (progress/CLI), pytest (tests).

---

## Prerequisites

Run from project root: `/home/zk/Desktop/cook/main/cdli/ottominer/`

Install updated deps before starting:
```bash
pip install chardet durak-nlp scikit-learn plotly matplotlib
pip install surya-ocr  # optional Tier 2 OCR
```

Run existing tests to establish baseline:
```bash
pytest ottominer/tests/ -v --tb=short 2>&1 | head -60
```

---

## Task 1: Fix setup.py and requirements.txt

**Files:**
- Modify: `setup.py`
- Modify: `requirements.txt`

**Step 1: Update requirements.txt**

Replace the full file content:

```
# Core dependencies
rich>=10.0.0
pyyaml>=6.0.0
psutil>=5.9.0
pymupdf4llm>=0.0.17
chardet>=5.0.0
durak-nlp>=0.1.0
scikit-learn>=1.3.0
plotly>=5.0.0
matplotlib>=3.7.0

# Optional extras are declared in setup.py
# Install with: pip install ottominer[ocr], ottominer[ollama], ottominer[dev]

# Development dependencies (pip install ottominer[dev])
# black>=23.0.0
# isort>=5.12.0
# flake8>=6.0.0
# pytest>=7.0.0
# pytest-timeout>=2.1.0
# pytest-cov>=4.1.0
# reportlab>=4.0.0
```

**Step 2: Update setup.py**

Replace the full file:

```python
from setuptools import setup, find_packages

setup(
    name="ottominer",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    python_requires='>=3.8',
    install_requires=[
        'rich>=10.0.0',
        'pyyaml>=6.0.0',
        'psutil>=5.9.0',
        'pymupdf4llm>=0.0.17',
        'chardet>=5.0.0',
        'durak-nlp>=0.1.0',
        'scikit-learn>=1.3.0',
        'plotly>=5.0.0',
        'matplotlib>=3.7.0',
    ],
    extras_require={
        'ocr': ['surya-ocr>=0.4.0'],
        'ollama': ['ollama>=0.1.0'],
        'embeddings': ['sentence-transformers>=2.0.0'],
        'dev': [
            'black>=23.0.0',
            'isort>=5.12.0',
            'flake8>=6.0.0',
            'pytest>=7.0.0',
            'pytest-timeout>=2.1.0',
            'pytest-cov>=4.1.0',
            'reportlab>=4.0.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'ottominer=ottominer.cli.main:main',
        ],
    },
)
```

**Step 3: Verify the entry point resolves (will fail until cli/main.py exists — that is expected)**

```bash
pip install -e . 2>&1 | tail -5
```

Expected: installs cleanly. Running `ottominer` will fail with ModuleNotFoundError until Task 12.

**Step 4: Commit**

```bash
git add setup.py requirements.txt
git commit -m "chore: update deps and add console_scripts entry point"
```

---

## Task 2: Create Shared Data Classes

**Files:**
- Create: `ottominer/analyzers/__init__.py`
- Create: `ottominer/analyzers/base.py`
- Create: `ottominer/tests/test_analyzers.py` (initial skeleton)

These dataclasses are the contract between every analyzer. Define them first so all other tasks can import them.

**Step 1: Write the failing test**

Create `ottominer/tests/test_analyzers.py`:

```python
import pytest
from ottominer.analyzers.base import TokenizedDocument, AnalyzedDocument


class TestDataClasses:
    def test_tokenized_document_fields(self):
        doc = TokenizedDocument(
            source_path="test.pdf",
            raw_text="Osmanlı metni",
            tokens=["Osmanlı", "metni"],
            lemmas=["Osmanlı", "metin"],
            offsets=[(0, 7), (8, 13)],
            filtered_tokens=["Osmanlı", "metni"],
        )
        assert doc.source_path == "test.pdf"
        assert doc.tokens == ["Osmanlı", "metni"]
        assert len(doc.offsets) == 2

    def test_analyzed_document_defaults(self):
        from ottominer.analyzers.base import TokenizedDocument
        tok = TokenizedDocument(
            source_path="x.pdf",
            raw_text="test",
            tokens=["test"],
            lemmas=["test"],
            offsets=[(0, 4)],
            filtered_tokens=["test"],
        )
        doc = AnalyzedDocument(tokenized=tok)
        assert doc.semantic_labels == {}
        assert doc.morphology == {}
        assert doc.genre_scores == {}
        assert doc.register == "unknown"
        assert doc.similarity_vector is None
```

**Step 2: Run the test to verify it fails**

```bash
pytest ottominer/tests/test_analyzers.py::TestDataClasses -v
```

Expected: `FAILED` — `ModuleNotFoundError: No module named 'ottominer.analyzers.base'`

**Step 3: Create `ottominer/analyzers/base.py`**

```python
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional


@dataclass
class TokenizedDocument:
    source_path: str
    raw_text: str
    tokens: List[str]
    lemmas: List[str]
    offsets: List[Tuple[int, int]]
    filtered_tokens: List[str]


@dataclass
class AnalyzedDocument:
    tokenized: TokenizedDocument
    semantic_labels: Dict[str, List[str]] = field(default_factory=dict)
    morphology: Dict[str, List[str]] = field(default_factory=dict)
    genre_scores: Dict[str, float] = field(default_factory=dict)
    register: str = "unknown"
    similarity_vector: Optional[List[float]] = None
```

**Step 4: Create `ottominer/analyzers/__init__.py`**

```python
from .base import TokenizedDocument, AnalyzedDocument

__all__ = ["TokenizedDocument", "AnalyzedDocument"]
```

**Step 5: Run the test to verify it passes**

```bash
pytest ottominer/tests/test_analyzers.py::TestDataClasses -v
```

Expected: `PASSED`

**Step 6: Commit**

```bash
git add ottominer/analyzers/__init__.py ottominer/analyzers/base.py ottominer/tests/test_analyzers.py
git commit -m "feat: add TokenizedDocument and AnalyzedDocument dataclasses"
```

---

## Task 3: Implement the Tokenizer

**Files:**
- Create: `ottominer/analyzers/tokenizer.py`
- Modify: `ottominer/tests/test_analyzers.py` (add tokenizer tests)

The tokenizer wraps `durak` for mechanics and applies Ottoman-specific post-processing. It degrades gracefully if `durak` is not installed.

**Step 1: Add failing tests to `test_analyzers.py`**

Append to the existing file:

```python
class TestTokenizer:
    def test_tokenize_basic_text(self):
        from ottominer.analyzers.tokenizer import OttomanTokenizer
        tok = OttomanTokenizer()
        doc = tok.tokenize("test.pdf", "Osmanlı devleti büyük bir imparatorluktu")
        assert isinstance(doc.tokens, list)
        assert len(doc.tokens) > 0
        assert doc.source_path == "test.pdf"
        assert doc.raw_text == "Osmanlı devleti büyük bir imparatorluktu"

    def test_stopwords_are_filtered(self):
        from ottominer.analyzers.tokenizer import OttomanTokenizer
        tok = OttomanTokenizer()
        # "ve", "bir", "bu" are in stopwords.json
        doc = tok.tokenize("test.pdf", "ve bu bir deneme metnidir")
        # stopwords should be removed from filtered_tokens
        assert "ve" not in doc.filtered_tokens
        assert "bir" not in doc.filtered_tokens

    def test_turkish_characters_preserved(self):
        from ottominer.analyzers.tokenizer import OttomanTokenizer
        tok = OttomanTokenizer()
        doc = tok.tokenize("test.pdf", "şehir köylü ağa çelebi")
        joined = " ".join(doc.tokens)
        assert "ş" in joined or "ğ" in joined or "ç" in joined

    def test_empty_text_returns_empty_doc(self):
        from ottominer.analyzers.tokenizer import OttomanTokenizer
        tok = OttomanTokenizer()
        doc = tok.tokenize("test.pdf", "")
        assert doc.tokens == []
        assert doc.filtered_tokens == []

    def test_offsets_match_token_count(self):
        from ottominer.analyzers.tokenizer import OttomanTokenizer
        tok = OttomanTokenizer()
        doc = tok.tokenize("test.pdf", "Osmanlı metni örnek")
        assert len(doc.offsets) == len(doc.tokens)
```

**Step 2: Run to verify they fail**

```bash
pytest ottominer/tests/test_analyzers.py::TestTokenizer -v
```

Expected: `FAILED` — `ModuleNotFoundError: No module named 'ottominer.analyzers.tokenizer'`

**Step 3: Create `ottominer/analyzers/tokenizer.py`**

```python
import re
import json
import logging
from pathlib import Path
from typing import List, Tuple, Set

from .base import TokenizedDocument

logger = logging.getLogger(__name__)

# Try to import durak; fall back to regex if unavailable
try:
    from durak import clean_text as durak_clean, tokenize as durak_tokenize
    from durak import attach_detached_suffixes, Lemmatizer
    _DURAK_AVAILABLE = True
    _lemmatizer = Lemmatizer()
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


def _durak_pipeline(text: str) -> Tuple[List[str], List[str], List[Tuple[int, int]]]:
    """Run the durak pipeline. Returns (tokens, lemmas, offsets)."""
    cleaned = durak_clean(text)
    token_objects = durak_tokenize(cleaned)
    # durak tokens have .text and .offset attributes
    tokens = [t.text for t in token_objects]
    offsets = [(t.offset, t.offset + len(t.text)) for t in token_objects]
    # Best-effort lemmatization — Ottoman vocab not in durak dict falls back to surface form
    try:
        lemmas = [_lemmatizer.lemmatize(t) or t for t in tokens]
    except Exception:
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

    def __init__(self, extra_stopwords: Set[str] = None):
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
```

**Step 4: Run the tests to verify they pass**

```bash
pytest ottominer/tests/test_analyzers.py::TestTokenizer -v
```

Expected: All 5 `PASSED`. If durak is not installed, the regex fallback handles everything.

**Step 5: Commit**

```bash
git add ottominer/analyzers/tokenizer.py ottominer/tests/test_analyzers.py
git commit -m "feat: implement OttomanTokenizer with durak backend and regex fallback"
```

---

## Task 4: Implement the Semantic Analyzer

**Files:**
- Modify: `ottominer/analyzers/semantic.py` (replace 8-line stub)
- Modify: `ottominer/tests/test_analyzers.py` (add semantic tests)

**Step 1: Add failing tests**

Append to `test_analyzers.py`:

```python
class TestSemanticAnalyzer:
    def test_labels_known_religious_token(self):
        from ottominer.analyzers.semantic import SemanticAnalyzer
        sa = SemanticAnalyzer()
        labels = sa.label_token("namaz")
        assert "religious" in labels

    def test_labels_known_economic_token(self):
        from ottominer.analyzers.semantic import SemanticAnalyzer
        sa = SemanticAnalyzer()
        labels = sa.label_token("ticaret")
        assert "economic" in labels

    def test_unknown_token_returns_empty(self):
        from ottominer.analyzers.semantic import SemanticAnalyzer
        sa = SemanticAnalyzer()
        labels = sa.label_token("xyznonexistent")
        assert labels == []

    def test_analyze_document_returns_analyzed(self):
        from ottominer.analyzers.semantic import SemanticAnalyzer
        from ottominer.analyzers.base import TokenizedDocument, AnalyzedDocument
        sa = SemanticAnalyzer()
        tok = TokenizedDocument(
            source_path="x.pdf",
            raw_text="namaz ticaret",
            tokens=["namaz", "ticaret"],
            lemmas=["namaz", "ticaret"],
            offsets=[(0, 5), (6, 13)],
            filtered_tokens=["namaz", "ticaret"],
        )
        doc = AnalyzedDocument(tokenized=tok)
        result = sa.analyze(doc)
        assert "namaz" in result.semantic_labels
        assert "religious" in result.semantic_labels["namaz"]

    def test_aggregate_stats(self):
        from ottominer.analyzers.semantic import SemanticAnalyzer
        from ottominer.analyzers.base import TokenizedDocument, AnalyzedDocument
        sa = SemanticAnalyzer()
        tok = TokenizedDocument(
            source_path="x.pdf",
            raw_text="namaz ticaret",
            tokens=["namaz", "ticaret"],
            lemmas=["namaz", "ticaret"],
            offsets=[(0, 5), (6, 13)],
            filtered_tokens=["namaz", "ticaret"],
        )
        doc = AnalyzedDocument(tokenized=tok)
        result = sa.analyze(doc)
        stats = sa.aggregate_stats(result)
        assert "category_counts" in stats
        assert stats["category_counts"].get("religious", 0) >= 1
```

**Step 2: Run to verify they fail**

```bash
pytest ottominer/tests/test_analyzers.py::TestSemanticAnalyzer -v
```

Expected: `FAILED` — ImportError or AttributeError from the stub.

**Step 3: Replace `ottominer/analyzers/semantic.py`**

```python
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
```

**Step 4: Update `ottominer/analyzers/__init__.py`**

```python
from .base import TokenizedDocument, AnalyzedDocument
from .tokenizer import OttomanTokenizer
from .semantic import SemanticAnalyzer

__all__ = ["TokenizedDocument", "AnalyzedDocument", "OttomanTokenizer", "SemanticAnalyzer"]
```

**Step 5: Run the tests**

```bash
pytest ottominer/tests/test_analyzers.py::TestSemanticAnalyzer -v
```

Expected: All 5 `PASSED`.

**Step 6: Commit**

```bash
git add ottominer/analyzers/semantic.py ottominer/analyzers/__init__.py ottominer/tests/test_analyzers.py
git commit -m "feat: implement SemanticAnalyzer with Ottoman category labeling"
```

---

## Task 5: Implement the Morphology Analyzer

**Files:**
- Create: `ottominer/analyzers/morphology.py`
- Modify: `ottominer/tests/test_analyzers.py`

**Step 1: Add failing tests**

Append to `test_analyzers.py`:

```python
class TestMorphologyAnalyzer:
    def test_detects_plural_suffix(self):
        from ottominer.analyzers.morphology import MorphologyAnalyzer
        ma = MorphologyAnalyzer()
        result = ma.analyze_token("köyler")
        assert "plural" in result

    def test_detects_verbal_suffix(self):
        from ottominer.analyzers.morphology import MorphologyAnalyzer
        ma = MorphologyAnalyzer()
        result = ma.analyze_token("gelmek")
        assert "verbal" in result

    def test_unknown_token_returns_empty(self):
        from ottominer.analyzers.morphology import MorphologyAnalyzer
        ma = MorphologyAnalyzer()
        result = ma.analyze_token("xyzqrst")
        assert result == []

    def test_analyze_document_populates_morphology(self):
        from ottominer.analyzers.morphology import MorphologyAnalyzer
        from ottominer.analyzers.base import TokenizedDocument, AnalyzedDocument
        ma = MorphologyAnalyzer()
        tok = TokenizedDocument(
            source_path="x.pdf",
            raw_text="köyler gelmek",
            tokens=["köyler", "gelmek"],
            lemmas=["köy", "gel"],
            offsets=[(0, 6), (7, 13)],
            filtered_tokens=["köyler", "gelmek"],
        )
        doc = AnalyzedDocument(tokenized=tok)
        result = ma.analyze(doc)
        assert "köyler" in result.morphology or "gelmek" in result.morphology

    def test_aggregate_counts(self):
        from ottominer.analyzers.morphology import MorphologyAnalyzer
        from ottominer.analyzers.base import TokenizedDocument, AnalyzedDocument
        ma = MorphologyAnalyzer()
        tok = TokenizedDocument(
            source_path="x.pdf",
            raw_text="köyler",
            tokens=["köyler"],
            lemmas=["köy"],
            offsets=[(0, 6)],
            filtered_tokens=["köyler"],
        )
        doc = AnalyzedDocument(tokenized=tok)
        doc = ma.analyze(doc)
        stats = ma.aggregate_stats(doc)
        assert "suffix_type_counts" in stats
```

**Step 2: Run to verify they fail**

```bash
pytest ottominer/tests/test_analyzers.py::TestMorphologyAnalyzer -v
```

Expected: `FAILED`

**Step 3: Create `ottominer/analyzers/morphology.py`**

```python
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
# Sort suffixes longest-first for greedy matching
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
```

**Step 4: Update `ottominer/analyzers/__init__.py`**

```python
from .base import TokenizedDocument, AnalyzedDocument
from .tokenizer import OttomanTokenizer
from .semantic import SemanticAnalyzer
from .morphology import MorphologyAnalyzer

__all__ = [
    "TokenizedDocument",
    "AnalyzedDocument",
    "OttomanTokenizer",
    "SemanticAnalyzer",
    "MorphologyAnalyzer",
]
```

**Step 5: Run the tests**

```bash
pytest ottominer/tests/test_analyzers.py::TestMorphologyAnalyzer -v
```

Expected: All 5 `PASSED`.

**Step 6: Commit**

```bash
git add ottominer/analyzers/morphology.py ottominer/analyzers/__init__.py ottominer/tests/test_analyzers.py
git commit -m "feat: implement MorphologyAnalyzer with longest-match suffix stripping"
```

---

## Task 6: Implement the Genre Analyzer

**Files:**
- Create: `ottominer/analyzers/genre.py`
- Modify: `ottominer/tests/test_analyzers.py`

**Step 1: Add failing tests**

Append to `test_analyzers.py`:

```python
class TestGenreAnalyzer:
    def test_detects_formal_register(self):
        from ottominer.analyzers.genre import GenreAnalyzer
        from ottominer.analyzers.base import TokenizedDocument, AnalyzedDocument
        ga = GenreAnalyzer()
        tok = TokenizedDocument(
            source_path="x.pdf",
            raw_text="efendi paşa hazret",
            tokens=["efendi", "paşa", "hazret"],
            lemmas=["efendi", "paşa", "hazret"],
            offsets=[(0, 6), (7, 11), (12, 18)],
            filtered_tokens=["efendi", "paşa", "hazret"],
        )
        doc = AnalyzedDocument(tokenized=tok)
        result = ga.analyze(doc)
        assert result.register == "formal"

    def test_detects_informal_register(self):
        from ottominer.analyzers.genre import GenreAnalyzer
        from ottominer.analyzers.base import TokenizedDocument, AnalyzedDocument
        ga = GenreAnalyzer()
        tok = TokenizedDocument(
            source_path="x.pdf",
            raw_text="selam naber dostum",
            tokens=["selam", "naber", "dostum"],
            lemmas=["selam", "naber", "dost"],
            offsets=[(0, 5), (6, 11), (12, 18)],
            filtered_tokens=["selam", "naber", "dostum"],
        )
        doc = AnalyzedDocument(tokenized=tok)
        result = ga.analyze(doc)
        assert result.register == "informal"

    def test_empty_document_returns_unknown(self):
        from ottominer.analyzers.genre import GenreAnalyzer
        from ottominer.analyzers.base import TokenizedDocument, AnalyzedDocument
        ga = GenreAnalyzer()
        tok = TokenizedDocument(
            source_path="x.pdf", raw_text="", tokens=[],
            lemmas=[], offsets=[], filtered_tokens=[],
        )
        doc = AnalyzedDocument(tokenized=tok)
        result = ga.analyze(doc)
        assert result.register == "unknown"

    def test_genre_scores_populated(self):
        from ottominer.analyzers.genre import GenreAnalyzer
        from ottominer.analyzers.base import TokenizedDocument, AnalyzedDocument
        ga = GenreAnalyzer()
        tok = TokenizedDocument(
            source_path="x.pdf",
            raw_text="ferman berat hüküm",
            tokens=["ferman", "berat", "hüküm"],
            lemmas=["ferman", "berat", "hüküm"],
            offsets=[(0, 6), (7, 12), (13, 18)],
            filtered_tokens=["ferman", "berat", "hüküm"],
        )
        doc = AnalyzedDocument(tokenized=tok)
        result = ga.analyze(doc)
        assert isinstance(result.genre_scores, dict)
        assert len(result.genre_scores) > 0
```

**Step 2: Run to verify they fail**

```bash
pytest ottominer/tests/test_analyzers.py::TestGenreAnalyzer -v
```

Expected: `FAILED`

**Step 3: Create `ottominer/analyzers/genre.py`**

```python
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

_FORMAL_MARKERS: Set[str] = {m.lower() for m in _FORMALITY_DATA.get("formal_markers", [])}
_INFORMAL_MARKERS: Set[str] = {m.lower() for m in _FORMALITY_DATA.get("informal_markers", [])}

_PERSIAN_COMPOUNDS: Set[str] = {
    s.lower() for s in _GENRE_DATA.get("persian_compounds", [])
}
_ARABIC_PATTERNS: Set[str] = _flatten(_GENRE_DATA.get("arabic_patterns", {}))
_HONORIFICS: Set[str] = _flatten(_GENRE_DATA.get("honorifics", {}))
_OFFICIAL_DOCS: Set[str] = _flatten(
    {"od": _GENRE_DATA.get("legal_and_administrative", {}).get("official_documents", [])}
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
            ) / total,
            "arabic_pattern_density": sum(
                1 for t in tokens_lower if t in _ARABIC_PATTERNS
            ) / total,
            "honorific_density": sum(
                1 for t in tokens_lower if t in _HONORIFICS
            ) / total,
            "official_doc_density": sum(
                1 for t in tokens_lower if t in _OFFICIAL_DOCS
            ) / total,
            "formal_marker_count": formal_count,
            "informal_marker_count": informal_count,
        }

        return doc
```

**Step 4: Update `ottominer/analyzers/__init__.py`**

```python
from .base import TokenizedDocument, AnalyzedDocument
from .tokenizer import OttomanTokenizer
from .semantic import SemanticAnalyzer
from .morphology import MorphologyAnalyzer
from .genre import GenreAnalyzer

__all__ = [
    "TokenizedDocument",
    "AnalyzedDocument",
    "OttomanTokenizer",
    "SemanticAnalyzer",
    "MorphologyAnalyzer",
    "GenreAnalyzer",
]
```

**Step 5: Run the tests**

```bash
pytest ottominer/tests/test_analyzers.py::TestGenreAnalyzer -v
```

Expected: All 4 `PASSED`.

**Step 6: Commit**

```bash
git add ottominer/analyzers/genre.py ottominer/analyzers/__init__.py ottominer/tests/test_analyzers.py
git commit -m "feat: implement GenreAnalyzer for register and genre classification"
```

---

## Task 7: Implement the Similarity Analyzer

**Files:**
- Create: `ottominer/analyzers/similarity.py`
- Modify: `ottominer/extractors/parallel.py` (remove dead TfidfVectorizer import)
- Modify: `ottominer/tests/test_analyzers.py`

**Step 1: Add failing tests**

Append to `test_analyzers.py`:

```python
class TestSimilarityAnalyzer:
    def _make_docs(self, texts):
        from ottominer.analyzers.base import TokenizedDocument, AnalyzedDocument
        docs = []
        for i, text in enumerate(texts):
            tokens = text.split()
            tok = TokenizedDocument(
                source_path=f"doc{i}.pdf",
                raw_text=text,
                tokens=tokens,
                lemmas=tokens,
                offsets=[(0, len(t)) for t in tokens],
                filtered_tokens=tokens,
            )
            docs.append(AnalyzedDocument(tokenized=tok))
        return docs

    def test_similarity_matrix_shape(self):
        from ottominer.analyzers.similarity import SimilarityAnalyzer
        sa = SimilarityAnalyzer()
        docs = self._make_docs(["namaz ibadet din", "ticaret pazar mal", "namaz dua"])
        result = sa.compute_batch(docs)
        assert len(result) == 3
        assert all(v is not None for v in result)

    def test_identical_docs_have_high_similarity(self):
        from ottominer.analyzers.similarity import SimilarityAnalyzer
        import numpy as np
        sa = SimilarityAnalyzer()
        docs = self._make_docs(["namaz ibadet din", "namaz ibadet din"])
        result = sa.compute_batch(docs)
        # Both docs should have high cosine similarity to each other
        v0 = result[0].similarity_vector
        v1 = result[1].similarity_vector
        dot = sum(a * b for a, b in zip(v0, v1))
        norm0 = sum(x**2 for x in v0) ** 0.5
        norm1 = sum(x**2 for x in v1) ** 0.5
        cosine = dot / (norm0 * norm1) if norm0 and norm1 else 0
        assert cosine > 0.9

    def test_single_document_batch_does_not_crash(self):
        from ottominer.analyzers.similarity import SimilarityAnalyzer
        sa = SimilarityAnalyzer()
        docs = self._make_docs(["namaz ibadet"])
        result = sa.compute_batch(docs)
        assert len(result) == 1

    def test_similarity_vector_values_are_finite(self):
        from ottominer.analyzers.similarity import SimilarityAnalyzer
        import math
        sa = SimilarityAnalyzer()
        docs = self._make_docs(["namaz ibadet", "ticaret pazar"])
        result = sa.compute_batch(docs)
        for doc in result:
            if doc.similarity_vector:
                assert all(math.isfinite(v) for v in doc.similarity_vector)
```

**Step 2: Run to verify they fail**

```bash
pytest ottominer/tests/test_analyzers.py::TestSimilarityAnalyzer -v
```

Expected: `FAILED`

**Step 3: Create `ottominer/analyzers/similarity.py`**

```python
import logging
from typing import List, Optional

from sklearn.feature_extraction.text import TfidfVectorizer

from .base import AnalyzedDocument

logger = logging.getLogger(__name__)


class SimilarityAnalyzer:
    """
    Computes pairwise TF-IDF cosine similarity across a batch of documents.

    Operates on filtered_tokens (stopwords already removed). Each document
    receives a similarity_vector (its TF-IDF row) so downstream code can
    compute pairwise similarity on demand.

    Optional: pass use_embeddings=True to use sentence-transformers if installed.
    Falls back to TF-IDF silently if sentence-transformers is unavailable.
    """

    def __init__(self, use_embeddings: bool = False):
        self._use_embeddings = use_embeddings and self._embeddings_available()

    @staticmethod
    def _embeddings_available() -> bool:
        try:
            import sentence_transformers  # noqa: F401
            return True
        except ImportError:
            return False

    def compute_batch(self, docs: List[AnalyzedDocument]) -> List[AnalyzedDocument]:
        """
        Compute TF-IDF vectors for all documents in the batch.
        Populates similarity_vector on each AnalyzedDocument.
        Returns the same list with vectors attached.
        """
        if not docs:
            return docs

        if self._use_embeddings:
            return self._embed_batch(docs)

        return self._tfidf_batch(docs)

    def _tfidf_batch(self, docs: List[AnalyzedDocument]) -> List[AnalyzedDocument]:
        """Build TF-IDF matrix; store each row as the document's similarity_vector."""
        corpora = [" ".join(d.tokenized.filtered_tokens) for d in docs]

        # Handle edge case: single document or all-empty corpora
        non_empty = [c for c in corpora if c.strip()]
        if not non_empty:
            return docs

        try:
            vectorizer = TfidfVectorizer()
            matrix = vectorizer.fit_transform(corpora)
            dense = matrix.toarray()
            for i, doc in enumerate(docs):
                doc.similarity_vector = dense[i].tolist()
        except Exception as e:
            logger.error(f"TF-IDF computation failed: {e}")

        return docs

    def _embed_batch(self, docs: List[AnalyzedDocument]) -> List[AnalyzedDocument]:
        """Use sentence-transformers for embedding-based similarity vectors."""
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
            sentences = [" ".join(d.tokenized.filtered_tokens) for d in docs]
            embeddings = model.encode(sentences)
            for i, doc in enumerate(docs):
                doc.similarity_vector = embeddings[i].tolist()
        except Exception as e:
            logger.warning(f"Embedding failed ({e}), falling back to TF-IDF")
            return self._tfidf_batch(docs)
        return docs
```

**Step 4: Remove the dead sklearn import from `ottominer/extractors/parallel.py`**

Remove lines 3-4:
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
```

**Step 5: Update `ottominer/analyzers/__init__.py`**

```python
from .base import TokenizedDocument, AnalyzedDocument
from .tokenizer import OttomanTokenizer
from .semantic import SemanticAnalyzer
from .morphology import MorphologyAnalyzer
from .genre import GenreAnalyzer
from .similarity import SimilarityAnalyzer

__all__ = [
    "TokenizedDocument",
    "AnalyzedDocument",
    "OttomanTokenizer",
    "SemanticAnalyzer",
    "MorphologyAnalyzer",
    "GenreAnalyzer",
    "SimilarityAnalyzer",
]
```

**Step 6: Run the tests**

```bash
pytest ottominer/tests/test_analyzers.py::TestSimilarityAnalyzer -v
```

Expected: All 4 `PASSED`.

**Step 7: Run all analyzer tests to confirm nothing broke**

```bash
pytest ottominer/tests/test_analyzers.py -v
```

Expected: All tests pass.

**Step 8: Commit**

```bash
git add ottominer/analyzers/similarity.py ottominer/analyzers/__init__.py \
    ottominer/extractors/parallel.py ottominer/tests/test_analyzers.py
git commit -m "feat: implement SimilarityAnalyzer with TF-IDF; remove dead sklearn import from parallel.py"
```

---

## Task 8: Implement the Pipeline

**Files:**
- Create: `ottominer/pipeline.py`
- Create: `ottominer/tests/test_pipeline.py`

The Pipeline is the public API. It chains: extract → tokenize → analyze → output → (optional) visualize.

**Step 1: Create `ottominer/tests/test_pipeline.py`**

```python
import pytest
import json
from pathlib import Path
from reportlab.pdfgen import canvas


def make_pdf(path: Path, text: str = "Osmanlı devleti namaz ticaret efendi paşa"):
    c = canvas.Canvas(str(path))
    c.drawString(50, 750, text)
    c.save()
    return path


class TestPipeline:
    def test_pipeline_runs_on_single_pdf(self, tmp_path):
        from ottominer.pipeline import Pipeline
        pdf = make_pdf(tmp_path / "test.pdf")
        output_dir = tmp_path / "output"
        p = Pipeline(output_dir=output_dir)
        results = p.run([pdf])
        assert len(results) == 1

    def test_output_json_is_created(self, tmp_path):
        from ottominer.pipeline import Pipeline
        pdf = make_pdf(tmp_path / "test.pdf")
        output_dir = tmp_path / "output"
        p = Pipeline(output_dir=output_dir)
        p.run([pdf])
        json_files = list(output_dir.glob("*.json"))
        assert len(json_files) >= 1

    def test_output_json_has_expected_keys(self, tmp_path):
        from ottominer.pipeline import Pipeline
        pdf = make_pdf(tmp_path / "test.pdf")
        output_dir = tmp_path / "output"
        p = Pipeline(output_dir=output_dir)
        p.run([pdf])
        json_file = next(output_dir.glob("*.json"))
        data = json.loads(json_file.read_text())
        assert "source_path" in data
        assert "tokens" in data
        assert "semantic_labels" in data
        assert "register" in data

    def test_pipeline_with_multiple_pdfs(self, tmp_path):
        from ottominer.pipeline import Pipeline
        pdfs = [make_pdf(tmp_path / f"doc{i}.pdf", f"metin {i} namaz") for i in range(3)]
        output_dir = tmp_path / "output"
        p = Pipeline(output_dir=output_dir)
        results = p.run(pdfs)
        assert len(results) == 3

    def test_pipeline_skips_failed_extraction_gracefully(self, tmp_path):
        from ottominer.pipeline import Pipeline
        bad_pdf = tmp_path / "bad.pdf"
        bad_pdf.write_bytes(b"not a pdf")
        output_dir = tmp_path / "output"
        p = Pipeline(output_dir=output_dir)
        results = p.run([bad_pdf])
        # Should return empty list or list with None, not raise
        assert isinstance(results, list)

    def test_extract_only_mode(self, tmp_path):
        from ottominer.pipeline import Pipeline
        pdf = make_pdf(tmp_path / "test.pdf")
        output_dir = tmp_path / "output"
        p = Pipeline(output_dir=output_dir, analyzers=[])
        results = p.run([pdf])
        assert len(results) == 1

    def test_similarity_vectors_populated_in_batch(self, tmp_path):
        from ottominer.pipeline import Pipeline
        pdfs = [make_pdf(tmp_path / f"doc{i}.pdf", "namaz ibadet ticaret") for i in range(2)]
        output_dir = tmp_path / "output"
        p = Pipeline(output_dir=output_dir)
        results = p.run(pdfs)
        # At least one result should have a similarity vector
        vectors = [r.similarity_vector for r in results if r and r.similarity_vector]
        assert len(vectors) > 0
```

**Step 2: Run to verify they fail**

```bash
pytest ottominer/tests/test_pipeline.py -v
```

Expected: `FAILED` — `No module named 'ottominer.pipeline'`

**Step 3: Create `ottominer/pipeline.py`**

```python
import json
import logging
from pathlib import Path
from typing import List, Optional, Union

from .extractors.pdf import PDFExtractor
from .analyzers.tokenizer import OttomanTokenizer
from .analyzers.semantic import SemanticAnalyzer
from .analyzers.morphology import MorphologyAnalyzer
from .analyzers.genre import GenreAnalyzer
from .analyzers.similarity import SimilarityAnalyzer
from .analyzers.base import AnalyzedDocument, TokenizedDocument

logger = logging.getLogger(__name__)

_ALL_ANALYZERS = ["semantic", "morphology", "genre", "similarity"]


class Pipeline:
    """
    OttoMiner end-to-end pipeline.

    Stages:
      1. Extract  — PDF to raw text via PDFExtractor (tiered, see extractors/pdf.py)
      2. Tokenize — Raw text to TokenizedDocument via OttomanTokenizer
      3. Analyze  — TokenizedDocument to AnalyzedDocument via selected analyzers
      4. Output   — Serialize results to JSON (and CSV summary if requested)

    Usage:
        p = Pipeline(output_dir=Path("output"))
        results = p.run(pdf_paths)

    The analyzers parameter controls which analyzers run. Pass an empty list
    to run extraction only. Defaults to all analyzers.
    """

    def __init__(
        self,
        output_dir: Union[str, Path] = "output",
        analyzers: Optional[List[str]] = None,
        extractor_config: Optional[dict] = None,
        use_embeddings: bool = False,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.analyzers = _ALL_ANALYZERS if analyzers is None else analyzers
        self._extractor = PDFExtractor(config=extractor_config or {})
        self._tokenizer = OttomanTokenizer()

        self._semantic = SemanticAnalyzer() if "semantic" in self.analyzers else None
        self._morphology = MorphologyAnalyzer() if "morphology" in self.analyzers else None
        self._genre = GenreAnalyzer() if "genre" in self.analyzers else None
        self._similarity = (
            SimilarityAnalyzer(use_embeddings=use_embeddings)
            if "similarity" in self.analyzers
            else None
        )

    def run(self, pdf_paths: List[Union[str, Path]]) -> List[Optional[AnalyzedDocument]]:
        """
        Run the full pipeline on a list of PDF paths.

        Returns a list of AnalyzedDocument objects (same length as input).
        Failed documents are represented as None and logged.
        """
        pdf_paths = [Path(p) for p in pdf_paths]
        analyzed: List[Optional[AnalyzedDocument]] = []

        for path in pdf_paths:
            doc = self._process_one(path)
            analyzed.append(doc)

        # Similarity requires a batch pass after all documents are tokenized
        if self._similarity and analyzed:
            valid = [d for d in analyzed if d is not None]
            if valid:
                self._similarity.compute_batch(valid)

        # Serialize to disk
        for doc in analyzed:
            if doc is not None:
                self._save_json(doc)

        return analyzed

    def _process_one(self, path: Path) -> Optional[AnalyzedDocument]:
        """Extract, tokenize, and analyze one document. Returns None on failure."""
        # Stage 1: Extract
        try:
            raw_text = self._extractor.extract(path)
            if not raw_text:
                logger.warning(f"Empty extraction result for {path}")
                raw_text = ""
        except Exception as e:
            logger.error(f"Extraction failed for {path}: {e}")
            return None

        # Stage 2: Tokenize
        try:
            tokenized: TokenizedDocument = self._tokenizer.tokenize(
                source_path=str(path),
                text=raw_text,
            )
        except Exception as e:
            logger.error(f"Tokenization failed for {path}: {e}")
            return None

        doc = AnalyzedDocument(tokenized=tokenized)

        # Stage 3: Analyze
        try:
            if self._semantic:
                self._semantic.analyze(doc)
            if self._morphology:
                self._morphology.analyze(doc)
            if self._genre:
                self._genre.analyze(doc)
        except Exception as e:
            logger.error(f"Analysis failed for {path}: {e}")

        return doc

    def _save_json(self, doc: AnalyzedDocument) -> Path:
        """Serialize an AnalyzedDocument to a JSON file in output_dir."""
        stem = Path(doc.tokenized.source_path).stem
        out_path = self.output_dir / f"{stem}.json"

        payload = {
            "source_path": doc.tokenized.source_path,
            "raw_text": doc.tokenized.raw_text,
            "tokens": doc.tokenized.tokens,
            "lemmas": doc.tokenized.lemmas,
            "filtered_tokens": doc.tokenized.filtered_tokens,
            "semantic_labels": doc.semantic_labels,
            "morphology": doc.morphology,
            "genre_scores": doc.genre_scores,
            "register": doc.register,
            "similarity_vector": doc.similarity_vector,
        }

        try:
            out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            logger.info(f"Saved analysis to {out_path}")
        except Exception as e:
            logger.error(f"Failed to save JSON for {doc.tokenized.source_path}: {e}")

        return out_path
```

**Step 4: Run the tests**

```bash
pytest ottominer/tests/test_pipeline.py -v
```

Expected: All 7 `PASSED`.

**Step 5: Run all tests to confirm nothing regressed**

```bash
pytest ottominer/tests/ -v --tb=short
```

Expected: All previously passing tests still pass.

**Step 6: Commit**

```bash
git add ottominer/pipeline.py ottominer/tests/test_pipeline.py
git commit -m "feat: implement Pipeline — extract→tokenize→analyze→output"
```

---

## Task 9: Extend PDF Extractor with Tiered Strategy

**Files:**
- Modify: `ottominer/extractors/pdf.py`
- Modify: `ottominer/tests/test_extractors.py`

Add tier detection logic. Tier 1 (pymupdf4llm) is always active. Tier 2 (Surya) is used when Tier 1 produces suspiciously low output.

**Step 1: Add failing tests to `test_extractors.py`**

Append:

```python
class TestTieredExtraction:
    def test_valid_text_pdf_uses_tier1(self, tmp_path, pdf_config):
        """Native text PDF should be handled by Tier 1 without escalation."""
        from ottominer.extractors.pdf import PDFExtractor
        pdf_path = create_test_pdf(tmp_path, "Osmanlı devleti büyük imparatorluk")
        extractor = PDFExtractor(pdf_config)
        result = extractor.extract(pdf_path)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_encoding_repair_does_not_crash(self, tmp_path, pdf_config):
        """Even if chardet repair path is triggered, no exception should escape."""
        from ottominer.extractors.pdf import PDFExtractor
        pdf_path = create_test_pdf(tmp_path, "Test content with encoding")
        extractor = PDFExtractor(pdf_config)
        result = extractor.extract(pdf_path)
        assert result is not None

    def test_image_pdf_flag_returns_string(self, tmp_path, pdf_config):
        """_is_image_based returns bool, not exception."""
        from ottominer.extractors.pdf import PDFExtractor
        pdf_path = create_test_pdf(tmp_path)
        extractor = PDFExtractor(pdf_config)
        result = extractor._is_image_based("some text content " * 10, pages=1)
        assert isinstance(result, bool)

    def test_encoding_issue_detection(self, tmp_path, pdf_config):
        """_has_encoding_issues returns bool."""
        from ottominer.extractors.pdf import PDFExtractor
        extractor = PDFExtractor(pdf_config)
        assert extractor._has_encoding_issues("\ufffd" * 50 + "normal text") is True
        assert extractor._has_encoding_issues("clean normal text here") is False
```

**Step 2: Run to verify they fail**

```bash
pytest ottominer/tests/test_extractors.py::TestTieredExtraction -v
```

Expected: `FAILED` on `_is_image_based` and `_has_encoding_issues` not existing.

**Step 3: Add tier detection methods to `ottominer/extractors/pdf.py`**

Add the following methods to the `PDFExtractor` class (before `_save_output`):

```python
    # --- Tier detection helpers ---

    def _is_image_based(self, extracted_text: str, pages: int) -> bool:
        """
        Return True if the PDF appears to be image-based (scanned).
        Heuristic: fewer than 100 characters per page on average.
        """
        if pages == 0:
            return True
        return len(extracted_text) / pages < 100

    def _has_encoding_issues(self, text: str) -> bool:
        """
        Return True if the text has a high ratio of Unicode replacement characters,
        indicating a probable encoding problem.
        Threshold: more than 5% of characters are \ufffd.
        """
        if not text:
            return False
        replacement_count = text.count("\ufffd")
        return replacement_count / len(text) > 0.05

    def _repair_encoding(self, file_path: Path) -> str:
        """
        Attempt to re-extract the PDF after detecting the encoding with chardet.
        Falls back to the original extraction if chardet fails.
        """
        try:
            import chardet
            with open(file_path, "rb") as f:
                raw = f.read()
            detected = chardet.detect(raw)
            encoding = detected.get("encoding") or "utf-8"
            logger.info(f"chardet detected encoding {encoding} for {file_path}")
            # Re-attempt extraction (pymupdf4llm handles encoding internally;
            # logging here for audit purposes)
            return pymupdf4llm.to_markdown(str(file_path), **self.pdf_config)
        except Exception as e:
            logger.warning(f"Encoding repair failed for {file_path}: {e}")
            return ""

    def _extract_with_surya(self, file_path: Path) -> str:
        """
        Tier 2: Extract using Surya OCR if installed.
        Returns empty string if Surya is not available.
        """
        try:
            from surya.recognition import batch_recognition
            from surya.model.recognition.model import load_model
            from surya.model.recognition.processor import load_processor
            import fitz  # pymupdf

            doc = fitz.open(str(file_path))
            images = [page.get_pixmap().tobytes("png") for page in doc]

            model = load_model()
            processor = load_processor()
            predictions = batch_recognition(images, [["tr"]] * len(images), model, processor)
            pages_text = [p.text_lines for p in predictions]
            return "\n\n".join(
                "\n".join(line.text for line in page) for page in pages_text
            )
        except ImportError:
            logger.warning(
                "Surya not installed. Install with: pip install ottominer[ocr]\n"
                "Returning empty string for image-based PDF."
            )
            return ""
        except Exception as e:
            logger.error(f"Surya extraction failed for {file_path}: {e}")
            return ""
```

Also update `_convert_to_markdown` to use the tier detection:

```python
    def _convert_to_markdown(self, file_path: Path) -> str:
        """Convert PDF to markdown using tiered extraction strategy."""
        if not self._is_valid_pdf(file_path):
            raise ValueError(f"Invalid or corrupted PDF file: {file_path}")

        # Tier 1: pymupdf4llm
        try:
            text = pymupdf4llm.to_markdown(str(file_path), **self.pdf_config)
        except Exception as e:
            logger.error(f"Tier 1 extraction failed for {file_path}: {e}")
            text = ""

        # Check for encoding issues and attempt repair
        if text and self._has_encoding_issues(text):
            logger.info(f"Encoding issues detected in {file_path}, attempting repair")
            repaired = self._repair_encoding(file_path)
            if repaired:
                text = repaired

        # Check if document is image-based (count pages via pymupdf)
        try:
            import fitz
            doc = fitz.open(str(file_path))
            pages = len(doc)
        except Exception:
            pages = max(1, len(text) // 3000)  # rough estimate

        if self._is_image_based(text, pages):
            logger.info(f"Image-based PDF detected: {file_path}, escalating to Tier 2")
            surya_text = self._extract_with_surya(file_path)
            if surya_text:
                text = surya_text

        if not text:
            logger.warning(f"All extraction tiers returned empty for {file_path}")

        return text
```

**Step 4: Run the new tests**

```bash
pytest ottominer/tests/test_extractors.py::TestTieredExtraction -v
```

Expected: All 4 `PASSED`.

**Step 5: Run all extractor tests**

```bash
pytest ottominer/tests/test_extractors.py -v
```

Expected: All pass.

**Step 6: Commit**

```bash
git add ottominer/extractors/pdf.py ottominer/tests/test_extractors.py
git commit -m "feat: add tiered extraction (Tier 1 pymupdf4llm, Tier 2 Surya) with encoding repair"
```

---

## Task 10: Implement HTML Report Visualizer

**Files:**
- Create: `ottominer/visualizers/__init__.py`
- Create: `ottominer/visualizers/html_report.py`
- Create: `ottominer/tests/test_visualizers.py`

**Step 1: Create `ottominer/tests/test_visualizers.py`**

```python
import pytest
import json
from pathlib import Path


def make_sample_json(output_dir: Path, stem: str = "doc0") -> Path:
    """Write a minimal analysis JSON file for visualizer tests."""
    data = {
        "source_path": f"{stem}.pdf",
        "raw_text": "namaz ticaret efendi",
        "tokens": ["namaz", "ticaret", "efendi"],
        "lemmas": ["namaz", "ticaret", "efendi"],
        "filtered_tokens": ["namaz", "ticaret"],
        "semantic_labels": {"namaz": ["religious"], "ticaret": ["economic"]},
        "morphology": {"ticaret": ["derivational"]},
        "genre_scores": {
            "persian_compound_density": 0.1,
            "arabic_pattern_density": 0.2,
            "honorific_density": 0.05,
            "official_doc_density": 0.0,
            "formal_marker_count": 2,
            "informal_marker_count": 0,
        },
        "register": "formal",
        "similarity_vector": [0.1, 0.3, 0.5],
    }
    path = output_dir / f"{stem}.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path


class TestHTMLReport:
    def test_html_report_creates_file(self, tmp_path):
        from ottominer.visualizers.html_report import HTMLReport
        make_sample_json(tmp_path)
        report = HTMLReport(input_dir=tmp_path, output_dir=tmp_path)
        out = report.generate()
        assert out.exists()
        assert out.suffix == ".html"

    def test_html_report_is_not_empty(self, tmp_path):
        from ottominer.visualizers.html_report import HTMLReport
        make_sample_json(tmp_path)
        report = HTMLReport(input_dir=tmp_path, output_dir=tmp_path)
        out = report.generate()
        content = out.read_text(encoding="utf-8")
        assert len(content) > 100

    def test_html_report_contains_plotly(self, tmp_path):
        from ottominer.visualizers.html_report import HTMLReport
        make_sample_json(tmp_path)
        report = HTMLReport(input_dir=tmp_path, output_dir=tmp_path)
        out = report.generate()
        content = out.read_text(encoding="utf-8")
        assert "plotly" in content.lower() or "<div" in content.lower()

    def test_html_report_with_multiple_docs(self, tmp_path):
        from ottominer.visualizers.html_report import HTMLReport
        for i in range(3):
            make_sample_json(tmp_path, stem=f"doc{i}")
        report = HTMLReport(input_dir=tmp_path, output_dir=tmp_path)
        out = report.generate()
        assert out.exists()
```

**Step 2: Run to verify they fail**

```bash
pytest ottominer/tests/test_visualizers.py::TestHTMLReport -v
```

Expected: `FAILED`

**Step 3: Create `ottominer/visualizers/__init__.py`**

```python
from .html_report import HTMLReport
from .static_figures import StaticFigures

__all__ = ["HTMLReport", "StaticFigures"]
```

**Step 4: Create `ottominer/visualizers/html_report.py`**

```python
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Union

import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)


def _load_jsons(input_dir: Path) -> List[Dict]:
    """Load all JSON analysis files from input_dir."""
    docs = []
    for path in sorted(input_dir.glob("*.json")):
        try:
            docs.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception as e:
            logger.warning(f"Could not load {path}: {e}")
    return docs


class HTMLReport:
    """
    Generates a self-contained interactive HTML report from pipeline JSON output.

    Charts included:
    - Token frequency (top 20 tokens, bar chart)
    - Semantic category distribution (donut chart)
    - Register breakdown per document (horizontal bar)
    - Suffix type frequency (bar chart)
    - Entropy curve (token frequency log-rank)
    """

    def __init__(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        report_name: str = "ottominer_report.html",
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.report_name = report_name

    def generate(self) -> Path:
        """Generate the HTML report. Returns the path to the saved file."""
        docs = _load_jsons(self.input_dir)
        if not docs:
            logger.warning(f"No JSON files found in {self.input_dir}")

        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                "Token Frequency (Top 20)",
                "Semantic Category Distribution",
                "Register per Document",
                "Suffix Type Frequency",
                "Entropy Curve (Log-Rank)",
                "",
            ],
            specs=[
                [{"type": "bar"}, {"type": "pie"}],
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "scatter"}, {"type": "scatter"}],
            ],
        )

        self._add_token_frequency(fig, docs, row=1, col=1)
        self._add_semantic_distribution(fig, docs, row=1, col=2)
        self._add_register_breakdown(fig, docs, row=2, col=1)
        self._add_suffix_frequency(fig, docs, row=2, col=2)
        self._add_entropy_curve(fig, docs, row=3, col=1)

        fig.update_layout(
            title_text="OttoMiner Analysis Report",
            height=1400,
            showlegend=True,
            font=dict(family="Arial", size=12),
        )

        out_path = self.output_dir / self.report_name
        fig.write_html(str(out_path), include_plotlyjs="cdn")
        logger.info(f"HTML report saved to {out_path}")
        return out_path

    # --- Chart builders ---

    def _add_token_frequency(self, fig, docs: List[Dict], row: int, col: int):
        counts: Dict[str, int] = defaultdict(int)
        for doc in docs:
            for token in doc.get("filtered_tokens", []):
                counts[token] += 1
        top = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:20]
        if top:
            tokens, freqs = zip(*top)
            fig.add_trace(go.Bar(x=list(tokens), y=list(freqs), name="Token Freq"), row=row, col=col)

    def _add_semantic_distribution(self, fig, docs: List[Dict], row: int, col: int):
        cat_counts: Dict[str, int] = defaultdict(int)
        for doc in docs:
            for cats in doc.get("semantic_labels", {}).values():
                for cat in cats:
                    cat_counts[cat] += 1
        if cat_counts:
            fig.add_trace(
                go.Pie(labels=list(cat_counts.keys()), values=list(cat_counts.values()),
                       name="Semantic"),
                row=row, col=col,
            )

    def _add_register_breakdown(self, fig, docs: List[Dict], row: int, col: int):
        names = [Path(d.get("source_path", f"doc{i}")).stem for i, d in enumerate(docs)]
        registers = [d.get("register", "unknown") for d in docs]
        register_map = {"formal": 1, "informal": -1, "mixed": 0, "unknown": 0}
        values = [register_map.get(r, 0) for r in registers]
        if names:
            fig.add_trace(
                go.Bar(x=values, y=names, orientation="h", name="Register",
                       marker_color=["green" if v > 0 else "red" if v < 0 else "gray" for v in values]),
                row=row, col=col,
            )

    def _add_suffix_frequency(self, fig, docs: List[Dict], row: int, col: int):
        type_counts: Dict[str, int] = defaultdict(int)
        for doc in docs:
            for types in doc.get("morphology", {}).values():
                for t in types:
                    type_counts[t] += 1
        if type_counts:
            sorted_items = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)
            labels, freqs = zip(*sorted_items)
            fig.add_trace(go.Bar(x=list(labels), y=list(freqs), name="Suffix Types"), row=row, col=col)

    def _add_entropy_curve(self, fig, docs: List[Dict], row: int, col: int):
        import math
        counts: Dict[str, int] = defaultdict(int)
        for doc in docs:
            for token in doc.get("filtered_tokens", []):
                counts[token] += 1
        sorted_freqs = sorted(counts.values(), reverse=True)
        if sorted_freqs:
            total = sum(sorted_freqs)
            probs = [f / total for f in sorted_freqs]
            entropy_vals = [
                -sum(p * math.log2(p) for p in probs[:k] if p > 0)
                for k in range(1, min(len(probs) + 1, 101))
            ]
            fig.add_trace(
                go.Scatter(x=list(range(1, len(entropy_vals) + 1)), y=entropy_vals,
                           mode="lines", name="Entropy"),
                row=row, col=col,
            )
```

**Step 5: Run the tests**

```bash
pytest ottominer/tests/test_visualizers.py::TestHTMLReport -v
```

Expected: All 4 `PASSED`.

**Step 6: Commit**

```bash
git add ottominer/visualizers/__init__.py ottominer/visualizers/html_report.py \
    ottominer/tests/test_visualizers.py
git commit -m "feat: implement HTMLReport visualizer with plotly charts"
```

---

## Task 11: Implement Static Figures Visualizer

**Files:**
- Create: `ottominer/visualizers/static_figures.py`
- Modify: `ottominer/tests/test_visualizers.py`

**Step 1: Add failing tests**

Append to `test_visualizers.py`:

```python
class TestStaticFigures:
    def test_static_figures_creates_png_files(self, tmp_path):
        from ottominer.visualizers.static_figures import StaticFigures
        make_sample_json(tmp_path)
        sf = StaticFigures(input_dir=tmp_path, output_dir=tmp_path)
        paths = sf.generate()
        png_files = [p for p in paths if p.suffix == ".png"]
        assert len(png_files) > 0

    def test_static_figures_files_are_nonzero(self, tmp_path):
        from ottominer.visualizers.static_figures import StaticFigures
        make_sample_json(tmp_path)
        sf = StaticFigures(input_dir=tmp_path, output_dir=tmp_path)
        paths = sf.generate()
        for p in paths:
            assert p.stat().st_size > 0

    def test_static_figures_respects_dpi(self, tmp_path):
        from ottominer.visualizers.static_figures import StaticFigures
        make_sample_json(tmp_path)
        sf = StaticFigures(input_dir=tmp_path, output_dir=tmp_path, dpi=72)
        paths = sf.generate()
        assert len(paths) > 0
```

**Step 2: Run to verify they fail**

```bash
pytest ottominer/tests/test_visualizers.py::TestStaticFigures -v
```

Expected: `FAILED`

**Step 3: Create `ottominer/visualizers/static_figures.py`**

```python
import json
import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Union

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/test use
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

_COLORBLIND_PALETTE = [
    "#0072B2", "#E69F00", "#56B4E9", "#009E73",
    "#F0E442", "#D55E00", "#CC79A7", "#999999",
]


def _load_jsons(input_dir: Path) -> List[Dict]:
    docs = []
    for path in sorted(input_dir.glob("*.json")):
        try:
            docs.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception as e:
            logger.warning(f"Could not load {path}: {e}")
    return docs


class StaticFigures:
    """
    Generates publication-ready matplotlib figures from pipeline JSON output.

    Outputs PNG files at the configured DPI. Uses a colorblind-safe palette.
    Each figure is saved as: <figure_name>.png
    """

    def __init__(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        dpi: int = 300,
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi

    def generate(self) -> List[Path]:
        """Generate all figures. Returns list of created file paths."""
        docs = _load_jsons(self.input_dir)
        paths = []
        paths += self._token_frequency(docs)
        paths += self._semantic_distribution(docs)
        paths += self._entropy_analysis(docs)
        paths += self._morphology_breakdown(docs)
        return paths

    def _save(self, fig, name: str) -> List[Path]:
        out = self.output_dir / f"{name}.png"
        try:
            fig.savefig(str(out), dpi=self.dpi, bbox_inches="tight")
            logger.info(f"Saved figure: {out}")
        except Exception as e:
            logger.error(f"Failed to save figure {name}: {e}")
        finally:
            plt.close(fig)
        return [out]

    def _token_frequency(self, docs: List[Dict]) -> List[Path]:
        counts: Dict[str, int] = defaultdict(int)
        for doc in docs:
            for t in doc.get("filtered_tokens", []):
                counts[t] += 1
        if not counts:
            return []
        top = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:20]
        tokens, freqs = zip(*top)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(tokens, freqs, color=_COLORBLIND_PALETTE[0])
        ax.set_xlabel("Token")
        ax.set_ylabel("Frequency")
        ax.set_title("Token Frequency (Top 20)")
        ax.tick_params(axis="x", rotation=45)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        return self._save(fig, "token_frequency")

    def _semantic_distribution(self, docs: List[Dict]) -> List[Path]:
        cat_counts: Dict[str, int] = defaultdict(int)
        for doc in docs:
            for cats in doc.get("semantic_labels", {}).values():
                for cat in cats:
                    cat_counts[cat] += 1
        if not cat_counts:
            return []
        labels = list(cat_counts.keys())
        values = list(cat_counts.values())
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.pie(values, labels=labels, colors=_COLORBLIND_PALETTE[:len(labels)],
               autopct="%1.1f%%", startangle=140)
        ax.set_title("Semantic Category Distribution")
        return self._save(fig, "semantic_distribution")

    def _entropy_analysis(self, docs: List[Dict]) -> List[Path]:
        counts: Dict[str, int] = defaultdict(int)
        for doc in docs:
            for t in doc.get("filtered_tokens", []):
                counts[t] += 1
        if not counts:
            return []
        sorted_freqs = sorted(counts.values(), reverse=True)
        total = sum(sorted_freqs)
        probs = [f / total for f in sorted_freqs]
        entropy_vals = [
            -sum(p * math.log2(p) for p in probs[:k] if p > 0)
            for k in range(1, min(len(probs) + 1, 101))
        ]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(range(1, len(entropy_vals) + 1), entropy_vals,
                color=_COLORBLIND_PALETTE[1], linewidth=1.5)
        ax.set_xlabel("Vocabulary rank")
        ax.set_ylabel("Cumulative entropy (bits)")
        ax.set_title("Entropy Curve")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        return self._save(fig, "entropy_analysis")

    def _morphology_breakdown(self, docs: List[Dict]) -> List[Path]:
        type_counts: Dict[str, int] = defaultdict(int)
        for doc in docs:
            for types in doc.get("morphology", {}).values():
                for t in types:
                    type_counts[t] += 1
        if not type_counts:
            return []
        sorted_items = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)
        labels, freqs = zip(*sorted_items)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(labels, freqs, color=_COLORBLIND_PALETTE[2])
        ax.set_xlabel("Suffix type")
        ax.set_ylabel("Count")
        ax.set_title("Morphological Suffix Type Breakdown")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        return self._save(fig, "morphology_breakdown")
```

**Step 4: Run the tests**

```bash
pytest ottominer/tests/test_visualizers.py::TestStaticFigures -v
```

Expected: All 3 `PASSED`.

**Step 5: Run all visualizer tests**

```bash
pytest ottominer/tests/test_visualizers.py -v
```

Expected: All pass.

**Step 6: Commit**

```bash
git add ottominer/visualizers/static_figures.py ottominer/tests/test_visualizers.py
git commit -m "feat: implement StaticFigures visualizer with matplotlib publication figures"
```

---

## Task 12: Wire the CLI

**Files:**
- Create: `ottominer/cli/main.py`
- Modify: `ottominer/cli/args.py`

**Step 1: Replace `ottominer/cli/args.py`**

```python
import argparse
import difflib
import re
from pathlib import Path
from typing import List


class CustomHelpFormatter(argparse.HelpFormatter):
    def __init__(self, prog):
        super().__init__(prog, max_help_position=50, width=100)


class SmartArgumentParser(argparse.ArgumentParser):
    def error(self, message: str):
        self.print_usage()
        valid_choices = self._get_valid_choices(message)
        if valid_choices:
            self.exit(
                2,
                f"{self.prog}: error: {message}\nDid you mean one of these?\n  "
                + "\n  ".join(valid_choices)
                + "\n",
            )
        else:
            self.exit(2, f"{self.prog}: error: {message}\n")

    def _get_valid_choices(self, error_message: str) -> List[str]:
        if "invalid choice" in error_message:
            match = re.search(
                r"argument .+: invalid choice: '(.+)' \(choose from (.+)\)",
                error_message,
            )
            if match:
                invalid = match.group(1)
                valid = match.group(2).replace("'", "").split(", ")
                return difflib.get_close_matches(invalid, valid, n=3, cutoff=0.6)
        return []


def build_parser() -> SmartArgumentParser:
    parser = SmartArgumentParser(
        prog="ottominer",
        description="OttoMiner — Ottoman Turkish NLP pipeline",
        formatter_class=CustomHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- extract ---
    ep = subparsers.add_parser("extract", help="Extract text from PDFs")
    ep.add_argument("--input", "-i", type=Path, required=True, help="Input directory of PDFs")
    ep.add_argument("--output", "-o", type=Path, required=True, help="Output directory")
    ep.add_argument("--ocr", action="store_true", help="Enable Surya OCR (Tier 2) if installed")
    ep.add_argument(
        "--ocr-backend",
        choices=["surya", "ollama", "mistral"],
        default="surya",
        help="OCR backend to use when --ocr is set",
    )
    ep.add_argument("--ocr-key", type=str, default=None, help="API key for cloud OCR backends")
    ep.add_argument("--workers", type=int, default=None, help="Worker count (default: cpu_count/2)")

    # --- analyze ---
    ap = subparsers.add_parser("analyze", help="Analyze extracted JSON output")
    ap.add_argument("--input", "-i", type=Path, required=True, help="Directory of JSON files from extract")
    ap.add_argument(
        "--analyzers",
        type=str,
        default="all",
        help="Comma-separated analyzers: semantic,morphology,genre,similarity (or 'all')",
    )
    ap.add_argument("--embeddings", action="store_true", help="Use sentence-transformers for similarity")

    # --- visualize ---
    vp = subparsers.add_parser("visualize", help="Generate reports from JSON output")
    vp.add_argument("--input", "-i", type=Path, required=True, help="Directory of JSON files")
    vp.add_argument("--output", "-o", type=Path, default=None, help="Output directory (default: same as input)")
    vp.add_argument("--html", action="store_true", help="Generate interactive HTML report")
    vp.add_argument("--figures", action="store_true", help="Generate static matplotlib figures")
    vp.add_argument("--dpi", type=int, default=300, help="DPI for static figures")

    # --- run (full pipeline) ---
    rp = subparsers.add_parser("run", help="Run full pipeline: extract + analyze + visualize")
    rp.add_argument("--input", "-i", type=Path, required=True, help="Input directory of PDFs")
    rp.add_argument("--output", "-o", type=Path, required=True, help="Output directory")
    rp.add_argument("--ocr", action="store_true")
    rp.add_argument("--ocr-backend", choices=["surya", "ollama", "mistral"], default="surya")
    rp.add_argument("--ocr-key", type=str, default=None)
    rp.add_argument(
        "--analyzers", type=str, default="all",
        help="Comma-separated analyzers or 'all'",
    )
    rp.add_argument("--embeddings", action="store_true")
    rp.add_argument("--html", action="store_true")
    rp.add_argument("--figures", action="store_true")
    rp.add_argument("--dpi", type=int, default=300)
    rp.add_argument("--workers", type=int, default=None)

    # --- legacy subcommands (deprecated) ---
    dp = subparsers.add_parser("data", help="[Deprecated] Use extract instead")
    dp.add_argument("--input", "-i", type=Path, required=True)
    dp.add_argument("--output", "-o", type=Path, required=True)
    dp.add_argument("--extraction-mode", choices=["simple", "ocr", "hybrid"], default="simple")
    dp.add_argument("--batch-size", type=int, default=100)
    dp.add_argument("--workers", type=int, default=1)

    lp = subparsers.add_parser("analysis", help="[Deprecated] Use analyze instead")
    lp.add_argument("--input", "-i", type=Path, required=True)
    lp.add_argument("--type", choices=["formality", "semantics", "genre", "morphology", "syntax"],
                    default="formality")

    return parser


def parse_args(args=None):
    return build_parser().parse_args(args)
```

**Step 2: Create `ottominer/cli/main.py`**

```python
import logging
import sys
from pathlib import Path

from .args import parse_args

logger = logging.getLogger(__name__)


def _resolve_analyzers(spec: str):
    all_analyzers = ["semantic", "morphology", "genre", "similarity"]
    if spec == "all":
        return all_analyzers
    selected = [a.strip() for a in spec.split(",")]
    unknown = [a for a in selected if a not in all_analyzers]
    if unknown:
        print(f"Unknown analyzers: {unknown}. Valid: {all_analyzers}", file=sys.stderr)
        sys.exit(1)
    return selected


def _get_pdf_paths(input_dir: Path):
    if not input_dir.is_dir():
        print(f"Input path is not a directory: {input_dir}", file=sys.stderr)
        sys.exit(1)
    paths = list(input_dir.glob("*.pdf")) + list(input_dir.glob("**/*.pdf"))
    if not paths:
        print(f"No PDF files found in {input_dir}", file=sys.stderr)
        sys.exit(1)
    return paths


def cmd_extract(args):
    from ..extractors.pdf import ParallelPDFExtractor
    import os, json
    pdfs = _get_pdf_paths(args.input)
    args.output.mkdir(parents=True, exist_ok=True)
    config = {}
    if args.workers:
        config["workers"] = args.workers
    extractor = ParallelPDFExtractor(config={"pdf_extraction": config})
    results = extractor.batch_extract(pdfs)
    for path_str, text in (results or {}).items():
        if text:
            stem = Path(path_str).stem
            out = args.output / f"{stem}.md"
            out.write_text(text, encoding="utf-8")
    print(f"Extracted {len([v for v in (results or {}).values() if v])} files to {args.output}")


def cmd_analyze(args):
    from ..visualizers.html_report import HTMLReport
    from ..visualizers.static_figures import StaticFigures
    # analyze command reads existing JSON from --input (already extracted)
    # runs analyzers and re-serializes — for now, just report what's there
    analyzers = _resolve_analyzers(args.analyzers)
    json_files = list(args.input.glob("*.json"))
    print(f"Found {len(json_files)} JSON files to re-analyze with: {analyzers}")
    # Full re-analysis from JSON is future work; existing output is usable as-is


def cmd_visualize(args):
    from ..visualizers.html_report import HTMLReport
    from ..visualizers.static_figures import StaticFigures
    out_dir = args.output or args.input
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.html and not args.figures:
        print("Specify --html and/or --figures", file=sys.stderr)
        sys.exit(1)
    if args.html:
        report = HTMLReport(input_dir=args.input, output_dir=out_dir)
        path = report.generate()
        print(f"HTML report: {path}")
    if args.figures:
        sf = StaticFigures(input_dir=args.input, output_dir=out_dir, dpi=args.dpi)
        paths = sf.generate()
        print(f"Static figures: {[str(p) for p in paths]}")


def cmd_run(args):
    from ..pipeline import Pipeline
    pdfs = _get_pdf_paths(args.input)
    analyzers = _resolve_analyzers(args.analyzers)
    p = Pipeline(
        output_dir=args.output,
        analyzers=analyzers,
        use_embeddings=args.embeddings,
    )
    results = p.run(pdfs)
    successful = [r for r in results if r is not None]
    print(f"Processed {len(successful)}/{len(pdfs)} documents. Output: {args.output}")

    if args.html or args.figures:
        from ..visualizers.html_report import HTMLReport
        from ..visualizers.static_figures import StaticFigures
        if args.html:
            report = HTMLReport(input_dir=args.output, output_dir=args.output)
            print(f"HTML report: {report.generate()}")
        if args.figures:
            sf = StaticFigures(input_dir=args.output, output_dir=args.output, dpi=args.dpi)
            print(f"Figures: {sf.generate()}")


_DISPATCH = {
    "extract": cmd_extract,
    "analyze": cmd_analyze,
    "visualize": cmd_visualize,
    "run": cmd_run,
    "data": cmd_extract,       # deprecated alias
    "analysis": cmd_analyze,   # deprecated alias
}


def main():
    args = parse_args()
    handler = _DISPATCH.get(args.command)
    if handler is None:
        print(f"Unknown command: {args.command}", file=sys.stderr)
        sys.exit(1)
    try:
        handler(args)
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as e:
        logger.exception(f"Command failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
```

**Step 3: Verify the CLI is callable**

```bash
pip install -e . -q && ottominer --help
```

Expected output:
```
usage: ottominer [-h] {extract,analyze,visualize,run,data,analysis} ...

OttoMiner — Ottoman Turkish NLP pipeline
...
```

**Step 4: Quick smoke test**

```bash
ottominer run --help
```

Expected: shows run subcommand options without errors.

**Step 5: Run the CLI tests**

```bash
pytest ottominer/tests/test_cli.py -v
```

Expected: Existing CLI tests pass.

**Step 6: Commit**

```bash
git add ottominer/cli/main.py ottominer/cli/args.py
git commit -m "feat: implement CLI entry point with extract/analyze/visualize/run subcommands"
```

---

## Task 13: Final Integration and Test Run

**Files:**
- No new files — verify everything works together.

**Step 1: Run the full test suite**

```bash
pytest ottominer/tests/ -v --tb=short 2>&1 | tee test_results.txt
```

Expected: All tests pass. Note any failures.

**Step 2: Run a full pipeline smoke test**

```bash
# Create a small test batch directory
mkdir -p /tmp/ottominer_smoke/pdfs

python3 -c "
from reportlab.pdfgen import canvas
for i in range(3):
    c = canvas.Canvas(f'/tmp/ottominer_smoke/pdfs/doc{i}.pdf')
    c.drawString(50, 750, 'Osmanlı devleti namaz ticaret efendi paşa ferman')
    c.drawString(50, 700, 'Tekke dergah cami ibadet sadaka kelam')
    c.save()
print('PDFs created')
"

ottominer run \
  --input /tmp/ottominer_smoke/pdfs \
  --output /tmp/ottominer_smoke/output \
  --html \
  --figures
```

Expected output:
```
Processed 3/3 documents. Output: /tmp/ottominer_smoke/output
HTML report: /tmp/ottominer_smoke/output/ottominer_report.html
Figures: ['.../token_frequency.png', '.../semantic_distribution.png', ...]
```

**Step 3: Verify output files**

```bash
ls /tmp/ottominer_smoke/output/
```

Expected: `doc0.json`, `doc1.json`, `doc2.json`, `ottominer_report.html`, `token_frequency.png`, `semantic_distribution.png`, `entropy_analysis.png`, `morphology_breakdown.png`

**Step 4: Run full test suite one final time**

```bash
pytest ottominer/tests/ --tb=short -q
```

Expected: All tests pass, no warnings about missing modules.

**Step 5: Final commit**

```bash
git add -A
git commit -m "feat: ottominer overhaul complete — full pipeline, tiered extraction, analyzers, visualizers, CLI"
```

---

## Summary

| Task | What it delivers |
|------|-----------------|
| 1 | setup.py entry point + updated deps |
| 2 | TokenizedDocument + AnalyzedDocument dataclasses |
| 3 | OttomanTokenizer (durak + regex fallback + Ottoman stopwords) |
| 4 | SemanticAnalyzer (Ottoman category labeling) |
| 5 | MorphologyAnalyzer (longest-match suffix stripping) |
| 6 | GenreAnalyzer (register + genre density scores) |
| 7 | SimilarityAnalyzer (TF-IDF + optional embeddings) |
| 8 | Pipeline (extract → tokenize → analyze → JSON output) |
| 9 | Tiered PDF extraction (encoding repair + Surya fallback) |
| 10 | HTMLReport visualizer (Plotly, self-contained) |
| 11 | StaticFigures visualizer (matplotlib, publication-ready) |
| 12 | CLI (extract/analyze/visualize/run subcommands + entry point) |
| 13 | Integration smoke test |
