# OttoMiner Enhancement Roadmap

## Phase 4: Additional OCR Backends

### Current State
- Tier 1: pymupdf4llm (text-native PDFs)
- Tier 2: chardet encoding repair
- Tier 3: Surya OCR (optional, CPU)
- Tier 4: Ollama vision models (optional, GPU)

### New Backends to Add
| Backend | Type | Strengths | Install |
|---------|------|-----------|---------|
| Tesseract | Local CPU | Fast, well-tested | `pip install pytesseract` |
| Mistral Vision | Cloud API | High accuracy, fast | `pip install mistralai` |
| Google Vision | Cloud API | Best OCR quality | `pip install google-cloud-vision` |
| Azure Document Intelligence | Cloud API | Layout preservation | `pip install azure-ai-formrecognizer` |
| PaddleOCR | Local GPU/CPU | Multilingual, fast | `pip install paddleocr` |
| EasyOCR | Local GPU/CPU | 80+ languages | `pip install easyocr` |

---

## Phase 5: Enhanced NLP Pipeline

### New Analyzers
| Analyzer | Purpose | Data Source |
|----------|---------|-------------|
| Lemmatizer | Word root extraction | turkish-stemmer, Zemberek |
| NER (Named Entity) | Person/place/org detection | Custom Ottoman gazetteers |
| Dependency Parser | Syntax analysis | stanza, transformers |
| Topic Modeler | Document clustering | LDA, BERTopic |
| Sentiment Analyzer | Formality/emotion scoring | Custom lexicon |

### New Features
- Word embeddings (fastText Turkish)
- Document similarity with semantic embeddings
- Cross-document coreference
- Temporal analysis for dated documents

---

## Phase 6: Enhanced Visualization

### New Chart Types
- Word clouds (Arabic script support)
- Topic distribution heatmap
- Document timeline
- Entity network graph
- Comparative analysis (side-by-side)
- PDF viewer with annotation overlay

### Export Formats
- LaTeX tables for publications
- PowerPoint/Keynote export
- Interactive Jupyter widgets
- JSON-LD for linked data

---

## Phase 7: Performance & Caching

### Caching Strategy
```
~/.ottominer/
├── cache/
│   ├── extractions/    # SHA256(pdf) → extracted text
│   ├── embeddings/     # Pre-computed TF-IDF vectors
│   └── models/         # Downloaded models
├── config.yaml
└── logs/
```

### Parallelization
- Multiprocessing for CPU-bound tasks
- GPU batching for OCR/inference
- Async I/O for cloud APIs
- Progress persistence (resume interrupted runs)

### Memory Optimization
- Streaming for large documents
- Lazy loading of analyzers
- Configurable batch sizes

---

## Implementation Priority

1. **Tesseract + EasyOCR** - No API keys needed, immediate value
2. **Extraction caching** - Big speedup for repeated runs
3. **Word embeddings** - Better similarity analysis
4. **Word clouds** - Quick visual win
5. **Mistral Vision** - Cloud fallback option
