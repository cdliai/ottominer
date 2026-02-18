# OttoMiner Overhaul Design

**Date:** 2026-02-18
**Approach:** B — Clean Architecture
**Status:** Approved

---

## 1. Context and Motivation

OttoMiner is an Ottoman Turkish NLP toolkit for extracting, processing, and analyzing archival documents, developed at BUCOLIN Lab, Bogazici University. It was published at SIGTURK 2024 (ACL, Bangkok).

The repository is at roughly 60% completeness. The PDF extraction pipeline is production-quality. The analysis layer is a skeleton — `analyzers/` is a dead folder, the CLI has no entry point, and several declared features have no implementation. The overhaul completes the vision: a tool that works as a CLI for researchers and as a Python package for developers, producing structured datasets and visual reports from Ottoman archival PDFs.

**Primary users:** researchers/historians and NLP engineers/developers.
**Primary outputs:** structured JSON/CSV datasets + HTML interactive reports + publication-ready matplotlib figures.

---

## 2. Goals

- Complete the analysis pipeline (semantic, morphological, genre, similarity)
- Replace the current Jaccard similarity with TF-IDF; add optional embedding-based similarity
- Integrate `durak` (Rust-backed Turkish NLP, `pip install durak-nlp`) as the tokenization engine with Ottoman-specific layering on top
- Build a tiered PDF extraction strategy that handles encoding issues and image-based PDFs without requiring cloud services
- Implement `visualizers/` for both HTML (interactive) and matplotlib (publication) output
- Wire everything into a working CLI with a proper `console_scripts` entry point
- Expand the test suite to cover all new modules

---

## 3. What Is Not in Scope

- Database-backed corpus manager
- Annotation interface or UI
- REST API
- Role-based access control
- Named entity recognition (separate future work)
- Interactive query/search system
- Ottoman-specific lemmatization in `durak` (tracked via upstream issue #148 — see Section 7)

---

## 4. Module Structure

### 4.1 Directory Layout

```
ottominer/
├── core/
│   ├── config.py              # consolidated config (absorbs scattered defaults)
│   ├── environment.py         # singleton env, directory management (minimal changes)
│   ├── data_manager.py        # unchanged
│   └── schema.py              # unchanged
│
├── extractors/
│   ├── base.py                # unchanged
│   └── pdf.py                 # extended with tiered extraction logic + chardet repair
│
├── analyzers/                 # REBUILT — the new analytical core
│   ├── __init__.py            # exports all analyzers, registers available names
│   ├── tokenizer.py           # durak wrapper + Ottoman-specific post-processing
│   ├── semantic.py            # Ottoman semantic labeling from semantics.json
│   ├── morphology.py          # suffix analysis via suffixes.json + durak lemmatizer
│   ├── genre.py               # genre/register classification from genre.json + formality.json
│   └── similarity.py          # TF-IDF default; optional embedding flag
│
├── pipeline.py                # NEW: Pipeline class — chains Extract→Tokenize→Analyze→Output
│
├── visualizers/               # NEW — replaces empty utils/visualization.py
│   ├── __init__.py
│   ├── html_report.py         # interactive Plotly HTML report
│   └── static_figures.py     # matplotlib publication figures
│
├── cli/
│   ├── main.py                # NEW: entry point, routes subcommands to Pipeline
│   ├── args.py                # updated with new subcommands (extract/analyze/visualize/run)
│   └── completion.py          # unchanged
│
├── utils/                     # unchanged
│   ├── logger.py
│   ├── cache.py
│   ├── decorators.py
│   ├── progress.py
│   ├── resources.py
│   └── cleanup.py
│
├── fdata/                     # unchanged (Ottoman lexicon JSON files)
│   ├── stopwords.json
│   ├── semantics.json
│   ├── suffixes.json
│   ├── genre.json
│   └── formality.json
│
└── tests/
    ├── conftest.py            # extended with new fixtures
    ├── test_core.py           # unchanged
    ├── test_extractors.py     # extended with tier detection tests
    ├── test_analyzers.py      # NEW: tokenizer, semantic, morphology, genre, similarity
    ├── test_pipeline.py       # NEW: end-to-end pipeline integration
    ├── test_visualizers.py    # NEW: HTML and figure output
    ├── test_cli.py            # extended with new subcommand tests
    ├── test_parallel.py       # unchanged
    ├── test_data_manager.py   # unchanged
    ├── test_integration.py    # extended
    └── test_utils.py          # unchanged
```

### 4.2 Key Architectural Principles

- **Analyzers are additive.** Each analyzer decorates a shared `AnalyzedDocument` dataclass without destroying prior results. Order does not matter.
- **Visualizers are pure consumers.** They read structured JSON output only. They never call extractors or analyzers. They can be run standalone on existing output.
- **Pipeline is the public API.** Both the CLI and programmatic users interact through `Pipeline`. Nothing below it is part of the public interface.
- **Tiers are optional installs.** Core deps stay minimal. OCR and local model support are extras.

---

## 5. End-to-End Pipeline

```
Input PDFs
    ↓
[1] EXTRACT     — tiered PDF extraction (see Section 6)
                  output: raw markdown text per document
    ↓
[2] TOKENIZE    — durak: clean → tokenize → attach suffixes → lemmatize → remove stopwords
                  Ottoman layer: merge durak stopwords with stopwords.json (174 terms)
                  output: TokenizedDocument dataclass
    ↓
[3] ANALYZE     — runs selected analyzers in parallel on TokenizedDocument:
                  • semantic.py   → tag tokens with semantic categories
                  • morphology.py → identify suffix types, case markers, verbal forms
                  • genre.py      → classify register, detect Persian/Arabic patterns
                  • similarity.py → pairwise TF-IDF similarity across documents
                  output: AnalyzedDocument dataclass (cumulative)
    ↓
[4] OUTPUT      — serialize to JSON (full data) and CSV (token/label summary)
                  saved to output/<batch_name>/
    ↓
[5] VISUALIZE   — reads output JSON, generates:
                  • HTML interactive report (Plotly)
                  • Static figures (matplotlib, optional --figures flag)
```

### 5.1 Data Structures

```python
@dataclass
class TokenizedDocument:
    source_path: str
    raw_text: str
    tokens: list[str]
    lemmas: list[str]
    offsets: list[tuple[int, int]]   # character-accurate, from durak
    filtered_tokens: list[str]       # stopwords removed

@dataclass
class AnalyzedDocument:
    tokenized: TokenizedDocument
    semantic_labels: dict[str, list[str]]    # token → [label, ...]
    morphology: dict[str, list[str]]         # token → [suffix_type, ...]
    genre_scores: dict[str, float]           # category → score
    register: str                            # "formal" | "informal"
    similarity_vector: list[float] | None   # populated after batch similarity pass
```

---

## 6. Extraction Strategy — Tiered

The PDF extractor auto-detects document type and escalates through installed tiers. Each tier is independent — if a tier is not installed, the pipeline logs a clear actionable warning and stops escalation there.

### Tier 1 — Always on (zero extra deps)

**Tool:** `pymupdf4llm` + `chardet`

**Triggers:** all PDFs by default.

**Logic:**
1. Attempt extraction with `pymupdf4llm`.
2. Count extractable characters. If `< 100 chars/page` on average → flag as image-based, escalate to Tier 2.
3. Check replacement character ratio (`\ufffd`). If `> 5%` of characters → encoding issue → run `chardet` to detect encoding, re-extract with detected encoding. If still broken → escalate to Tier 2.

**When it handles the case:** ~80% of Ottoman archival PDFs (text-native, may have encoding issues).

### Tier 2 — Optional OCR (recommended extra)

**Tool:** `surya`

**Install:** `pip install ottominer[ocr]`

**Triggers:** image-based pages or persistent encoding failures from Tier 1.

**Why Surya over Docling:** Surya is lighter, installs as a pure pip package, supports 90+ languages, and is layout-aware without requiring heavyweight HuggingFace model downloads. Docling's table/layout intelligence is unnecessary for the simple layouts in this corpus.

**Behavior:** Surya processes individual pages flagged by Tier 1. Mixed documents (some text pages, some image pages) are handled page-by-page — Tier 1 for text pages, Tier 2 for image pages.

### Tier 3 — Local Vision Model (for GPU users)

**Tools:** `deepseek-ocr` or `qwen2.5-vl` via Ollama

**Install:** `pip install ottominer[ollama]` + Ollama running locally with model pulled

**Triggers:** `--ocr-backend ollama` flag (explicit opt-in only, never auto-triggered)

**Default model:** `deepseek-ocr` (3B, 570M active parameters, token-efficient, fast on modest GPUs)

**Alternative:** `qwen2.5-vl:7b` (higher accuracy, more VRAM, beats Mistral-OCR on benchmarks)

**Use case:** bulk batch processing on a lab server with a GPU, or when Surya accuracy is insufficient.

### Tier 4 — Cloud API (last resort, explicit opt-in)

**Tool:** Mistral OCR API

**Triggers:** `--ocr-backend mistral --ocr-key <key>` (user must provide API key)

**Use case:** extremely degraded scans where local options fail. Not recommended as a default.

### Tier Summary

| Tier | Tool | Install | GPU | Auto-trigger | Use when |
|------|------|---------|-----|--------------|----------|
| 1 | pymupdf4llm + chardet | core | no | always | text-native PDFs |
| 2 | Surya | `[ocr]` | no | yes (on failure) | image pages, scans |
| 3 | DeepSeek-OCR / Qwen2.5-VL | `[ollama]` | recommended | manual flag | bulk jobs, GPU available |
| 4 | Mistral OCR API | `[cloud]` | no | manual flag | last resort |

---

## 7. Analyzer Layer

### 7.1 Tokenizer (`analyzers/tokenizer.py`)

Wraps `durak`'s `process_text()` pipeline:

```
durak.clean_text()           # Unicode normalization, Turkish İ/ı handling
durak.tokenize()             # regex-based, preserves Turkish morphology
durak.attach_detached_suffixes()
durak.Lemmatizer()           # tiered: lookup → heuristic → hybrid
durak.remove_stopwords()     # modern Turkish stopwords (durak built-in)
    +
Ottoman stopword filter      # our stopwords.json (174 terms) applied after durak
```

**Ottoman compatibility note:** `durak` targets modern Turkish. Its lemma dictionaries will produce incorrect or no results for Ottoman-specific vocabulary. The tokenizer treats durak's lemmatization as a best-effort baseline and does not propagate lemmas for tokens not found in the lookup table (falls back to surface form). A `language_variant` hook is tracked upstream at [cdliai/durak#148](https://github.com/cdliai/durak/issues/148) — when available, this tokenizer will pass `language_variant="ottoman"` to defer lemmatization to caller-supplied resources.

### 7.2 Semantic Analyzer (`analyzers/semantic.py`)

Loads `fdata/semantics.json` (12KB, categories: religious, cultural, political, economic).

For each token in `filtered_tokens`: look up against semantic label index. One token can carry multiple labels. Returns `dict[token, list[label]]`.

Aggregate stats per document: label frequency counts, dominant category, label coverage ratio (what fraction of content tokens have a semantic label).

### 7.3 Morphology Analyzer (`analyzers/morphology.py`)

Uses `fdata/suffixes.json` — case markers, possessive markers, verbal suffixes.

Strategy: try suffix stripping from longest to shortest match against the suffix table. Classify each token by what suffix type it carries (if any). This is rule-based, not statistical — fast and deterministic.

Returns per-token suffix classification and aggregate counts (e.g., genitive 23%, locative 11%, verbal 8%).

### 7.4 Genre Analyzer (`analyzers/genre.py`)

Uses `fdata/genre.json` (Persian compounds, Arabic patterns, honorifics) and `fdata/formality.json` (formal/informal markers).

Two scores per document:
- **Register score:** ratio of formal markers to informal markers → `"formal"` / `"informal"` / `"mixed"`
- **Genre signals:** presence and density of Persian compound patterns, Arabic loanword patterns, honorific forms

These signals map loosely to document genre (administrative, religious, literary, personal correspondence).

### 7.5 Similarity Analyzer (`analyzers/similarity.py`)

**Default (always available):** sklearn `TfidfVectorizer` on `filtered_tokens`.

- Operates at batch level — builds a corpus matrix across all documents in a run
- Returns pairwise cosine similarity matrix
- Stores each document's TF-IDF vector in `AnalyzedDocument.similarity_vector` for later use

**Optional (`--embeddings` flag):** `sentence-transformers` with a multilingual model. Triggered only if `sentence-transformers` is installed. Falls back to TF-IDF silently if not available.

Adds `sklearn` to core requirements (it was already imported but missing from `requirements.txt`). Removes the dead `TfidfVectorizer` import that currently exists in `extractors/parallel.py`.

---

## 8. Visualizer Layer

Visualizers are standalone consumers of the pipeline's JSON output. They can be invoked as part of a full pipeline run or independently via `ottominer visualize`.

### 8.1 HTML Report (`visualizers/html_report.py`)

**Library:** `plotly` (added to requirements)

**Output:** single self-contained `.html` file per batch (all assets inlined, no server needed)

**Contents:**
| Chart | Type | Data source |
|-------|------|-------------|
| Token frequency | bar chart | `filtered_tokens` counts |
| Semantic label distribution | donut chart | `semantic_labels` aggregate |
| Genre/register breakdown | horizontal bar | `genre_scores` per document |
| Suffix frequency | heatmap | `morphology` aggregate |
| Document similarity matrix | heatmap | `similarity_vector` matrix |
| Entropy curve | line chart | computed from token frequency distribution (replaces/extends `docs/comparative_entropy_analysis.html`) |

### 8.2 Static Figures (`visualizers/static_figures.py`)

**Library:** `matplotlib` (already a transitive dep, made explicit)

**Trigger:** `--figures` flag

**Output:** one `.pdf` and one `.png` per figure type, at 300 DPI by default (configurable)

**Figures exported:**
- `token_frequency.{pdf,png}`
- `semantic_distribution.{pdf,png}`
- `entropy_analysis.{pdf,png}`
- `similarity_matrix.{pdf,png}`
- `morphology_breakdown.{pdf,png}`

Style: minimal, publication-ready (no gridlines, clean axes, consistent font sizes). No color schemes that depend on screen rendering — uses colorblind-safe palettes.

---

## 9. CLI Design

### 9.1 Entry Point

`setup.py` gains:

```python
entry_points={
    'console_scripts': [
        'ottominer=ottominer.cli.main:main',
    ],
}
```

`cli/main.py` is the new file: parses top-level subcommand, instantiates `Pipeline`, delegates.

### 9.2 Subcommand Reference

```
ottominer extract   --input PATH  --output PATH  [--ocr] [--ocr-backend surya|ollama|mistral] [--ocr-key KEY]
ottominer analyze   --input PATH  [--analyzers semantic,morphology,genre,similarity | all] [--embeddings]
ottominer visualize --input PATH  [--html] [--figures] [--dpi INT]
ottominer run       --input PATH  --output PATH  [all extract+analyze+visualize flags composable]
```

`ottominer run` is the one-command full pipeline for researchers. `extract`, `analyze`, `visualize` are for developers who want to compose steps.

### 9.3 Backward Compatibility

The existing `data` and `analysis` subcommands in `args.py` are kept for now but marked deprecated in help text. They route to the new implementations internally.

---

## 10. Dependencies

### Core (`pip install ottominer`)

```
rich>=10.0.0
pymupdf4llm>=0.0.17
psutil>=5.9.0
pyyaml>=6.0.0
chardet>=5.0.0
durak-nlp>=0.1.0
scikit-learn>=1.3.0
plotly>=5.0.0
matplotlib>=3.7.0
```

### Optional extras

```
[ocr]       surya>=0.4.0
[ollama]    ollama>=0.1.0
[cloud]     (no extra package — Mistral uses requests, already transitive)
[embeddings] sentence-transformers>=2.0.0
[dev]       black, isort, flake8, pytest, pytest-cov, pytest-timeout, reportlab
```

Removes from requirements: `reportlab` moves to `[dev]` (only used in tests).

---

## 11. Testing Strategy

### Coverage targets

| Module | Current | Target |
|--------|---------|--------|
| extractors/ | ~80% | 90% |
| analyzers/ | ~0% | 85% |
| pipeline.py | 0% | 80% |
| visualizers/ | 0% | 75% |
| cli/ | ~50% | 80% |
| utils/ | ~70% | 80% |

### New test files

**`test_analyzers.py`**
- Tokenizer: Turkish character normalization, stopword merge (durak + Ottoman), offset correctness, graceful handling of unknown Ottoman vocabulary
- Semantic: label assignment from semantics.json, multi-label tokens, documents with zero semantic matches
- Morphology: suffix stripping correctness, longest-match priority, unknown tokens pass through clean
- Genre: formal/informal register scoring, Persian compound detection, empty document edge case
- Similarity: TF-IDF matrix shape, cosine values in [0,1], single-document batch (no crash), embeddings fallback

**`test_pipeline.py`**
- Full run on a small synthetic PDF batch (reportlab-generated in fixture)
- Partial runs: extract only, analyze only on pre-existing output, visualize only
- Tier escalation: mock pymupdf returning low char count, assert Surya called
- Output file presence and JSON schema validation

**`test_visualizers.py`**
- HTML report: output file exists, is valid HTML, contains expected chart div IDs
- Static figures: output `.png` files exist, non-zero file size, correct DPI metadata

---

## 12. Upstream Issue: durak Ottoman Turkish Support

Filed at: [cdliai/durak#148](https://github.com/cdliai/durak/issues/148)

**Request:** Add a `language_variant` parameter (`modern` | `ottoman` | `historical`) so callers can declare language mode. Ottoman mode would skip modern-Turkish lemma lookups and defer to caller-supplied dictionaries.

**Current workaround in this design:** The tokenizer uses durak for mechanics (tokenization, suffix attachment, Unicode normalization) and treats lemmatization as best-effort — surface form used as fallback for Ottoman vocabulary not in durak's modern dictionary. This is implemented with the `language_variant` extension point in mind so adoption of the upstream change is a clean one-line update.

**Integration plan for when the upstream issue is resolved:**
1. Update `durak-nlp` version pin
2. Pass `language_variant="ottoman"` in `tokenizer.py`
3. Supply `fdata/suffixes.json` and `fdata/stopwords.json` as custom resources to durak
4. Remove the manual Ottoman stopword post-filter (durak will handle it)

---

## 13. Implementation Order

The implementation should proceed in this order to ensure each layer is testable before the next is built on top of it:

1. Fix `setup.py` entry point and `cli/main.py` stub — makes the tool installable and runnable immediately
2. Rebuild `analyzers/__init__.py` and `analyzers/tokenizer.py` — foundation for all analysis
3. Implement `analyzers/semantic.py`, `morphology.py`, `genre.py` — the Ottoman intelligence layer
4. Implement `analyzers/similarity.py` — replaces dead Jaccard code, adds sklearn properly
5. Implement `pipeline.py` — wires extractors + analyzers + output serialization
6. Extend `extractors/pdf.py` with tier detection logic and chardet repair
7. Implement `visualizers/html_report.py` and `visualizers/static_figures.py`
8. Update `cli/args.py` with new subcommands, wire into `cli/main.py`
9. Expand test suite across all new modules
10. Update `setup.py` deps, `requirements.txt`, and install extras
