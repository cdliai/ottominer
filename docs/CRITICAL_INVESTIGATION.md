# OttoMiner: Critical Investigation Areas

## The #1 Challenge: Arabic Script OCR

### Problem
Most Ottoman documents are written in **Arabic script (elifba)**, not Latin.
Current backends (Tesseract, EasyOCR) have limited Arabic script support.

### Impact
- Without this, 90%+ of Ottoman corpus is inaccessible
- Current extraction produces garbled or empty text

### Solutions to Investigate

| Option | Accuracy | Speed | Cost | Effort |
|--------|----------|-------|------|--------|
| **Google Cloud Vision** | ★★★★★ | Fast | $$$ | Low |
| **Azure Document Intelligence** | ★★★★☆ | Fast | $$$ | Low |
| **Kraken OCR** | ★★★★☆ | Medium | Free | Medium |
| **Custom Transformer** | ★★★★★ | Slow | GPU | High |
| **Transkribus** | ★★★★★ | Medium | €€ | Low |

### Recommended Path
1. **Immediate**: Add Google Vision backend (best Arabic support)
2. **Medium-term**: Integrate Kraken (open-source, trainable)
3. **Long-term**: Fine-tune TrOCR on Ottoman manuscripts

---

## The #2 Challenge: Turkish Lemmatization

### Problem
Turkish is **agglutinative** - words can have 10+ suffixes.
Example: `gözlemekleştirilemezmiş` = "supposedly cannot be made observable"

Current: We detect suffix types but don't extract lemmas.

### Impact
- Token frequency is inflated (same word counted multiple times)
- Similarity analysis is noisy
- Vocabulary analysis is inaccurate

### Solutions

| Option | Accuracy | Coverage |
|--------|----------|----------|
| **Zemberek-NLP** | ★★★★★ | Modern Turkish |
| **TurkishStemmer** | ★★★☆☆ | Basic |
| **TurNeT** | ★★★★☆ | Good |
| **Custom Ottoman rules** | ★★★★☆ | Historical |

### Recommended Path
1. Integrate Zemberek-NLP (Java, has Python bindings)
2. Add Ottoman-specific morphological rules
3. Create historical Turkish lemma dictionary

---

## The #3 Challenge: Historical Document Preprocessing

### Problem
Scanned documents have:
- Paper texture and stains
- Bleed-through from back page
- Handwritten marginalia
- Damaged/torn pages
- Variable ink quality

### Current Gap
We pass raw PDF pages to OCR. Preprocessing would improve accuracy 20-50%.

### Preprocessing Pipeline Needed
```
Original → Binarization → Noise Removal → 
Deskewing → Contrast Enhancement → OCR
```

### Tools to Integrate
- **OpenCV** for image preprocessing
- **scikit-image** for binarization
- **historical-image-preprocessing** library

---

## The #4 Challenge: Named Entity Recognition

### Problem
Ottoman documents contain many named entities:
- Sultans, viziers, pashas
- Cities, regions, countries
- Religious titles, institutions
- Dates (Hijri calendar)

Current: No NER capability.

### Impact
- Cannot do prosopography (study of people)
- Cannot geolocate documents
- Cannot build temporal indexes

### Solutions
| Option | Training Data Needed |
|--------|---------------------|
| **spaCy custom model** | 10K+ annotated |
| **HuggingFace transformers** | 5K+ annotated |
| **Gazetteer + rules** | Dictionary only |

### Recommended Path
1. Build Ottoman entity gazetteer (sultans, cities, titles)
2. Rule-based NER first (fast, no training)
3. Annotate corpus for ML-based NER

---

## Priority Matrix

| Challenge | Impact | Effort | Priority |
|-----------|--------|--------|----------|
| Arabic Script OCR | Critical | Medium | **P0** |
| Lemmatization | High | Medium | **P1** |
| Document Preprocessing | High | Low | **P1** |
| NER | Medium | High | **P2** |
| Topic Modeling | Medium | Low | **P3** |
| Word Embeddings | Low | High | **P4** |

---

## Immediate Next Steps

1. **Add Google Vision OCR backend** (handles Arabic script)
2. **Add document preprocessing module** (improves all OCR)
3. **Integrate Turkish lemmatizer** (Zemberek or TurkishStemmer)
4. **Build Ottoman entity gazetteer** (people, places, titles)

---

## Research Questions

1. What % of your corpus is Arabic script vs Latin?
2. What time period? (15th-19th century affects orthography)
3. What document types? (fermans, court records, letters?)
4. Do you have any ground truth transcriptions for OCR evaluation?
5. What's the target use case? (search, NER, quantitative analysis?)
