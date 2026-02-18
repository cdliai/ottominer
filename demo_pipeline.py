#!/usr/bin/env python3
"""
OttoMiner Demo - Full pipeline test with timing and analytics.

Usage:
    python3 demo_pipeline.py
"""

import json
import time
import sys
from pathlib import Path
from datetime import datetime
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))


class Timer:
    def __init__(self, name):
        self.name = name
        self.start = None
        self.elapsed = None

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start


def create_sample_pdfs(output_dir: Path, count: int = 3):
    """Create sample PDFs with Ottoman Turkish text."""
    from reportlab.pdfgen import canvas
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    samples = [
        {
            "name": "religious_text",
            "lines": [
                "Osmanlı devletinde namaz ibadeti çok önemliydi.",
                "Cami ve mescitler her yerde inşa edildi.",
                "İmam ve müezzinler halkı namaza davet ederdi.",
                "Dini bayramlar büyük törenlerle kutlanırdı.",
                "Ramazan ayı oruç tutulur ve teravih namazları kılınırdı.",
                "Sadaka ve zekat vermek farzdı.",
                "Hac farizası için kervanlar düzenlenirdi.",
                "Kur'an kursları ve medreseler yaygındı.",
            ],
        },
        {
            "name": "administrative_text",
            "lines": [
                "Ferman padişah tarafından yazıldı.",
                "Vezir ve sadrazam devlet işlerini yönetti.",
                "Kadı ve müftü adaleti sağladı.",
                "Berat ve hüccet resmi belgelerdi.",
                "Mukataa ve timar sistemi uygulandı.",
                "Defterdar mali işlerden sorumluydu.",
                "Sancak beyleri eyaletleri yönetti.",
                "Divan-ı hümayun önemli kararları aldı.",
            ],
        },
        {
            "name": "literary_text",
            "lines": [
                "Şairler kasideler ve gazeller yazdı.",
                "Divan edebiyatı saray çevresinde gelişti.",
                "Mesnevi ve hikayeler anlatıldı.",
                "Aşık ve ozanlar halk hikayeleri anlattı.",
                "Gül ve bülbül motifleri sık kullanıldı.",
                "Aşk ve ayrılık temaları işlendi.",
                "Şairler maaş ve vakıf gelirleri aldı.",
                "Tezkireler şair biyografilerini içerdi.",
            ],
        },
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_paths = []

    for i in range(count):
        sample = samples[i % len(samples)]
        pdf_path = output_dir / f"{sample['name']}_{i}.pdf"

        c = canvas.Canvas(str(pdf_path))
        c.setFont("Helvetica", 12)

        y = 750
        c.drawString(50, y, f"Document: {sample['name'].replace('_', ' ').title()}")
        y -= 30

        for line in sample["lines"]:
            c.drawString(50, y, line)
            y -= 20
            if y < 100:
                c.showPage()
                c.setFont("Helvetica", 12)
                y = 750

        c.save()
        pdf_paths.append(pdf_path)

    return pdf_paths


def print_section(title):
    print(f"\n{'=' * 60}")
    print(f" {title}")
    print(f"{'=' * 60}")


def print_subsection(title):
    print(f"\n--- {title} ---")


def format_time(seconds):
    if seconds < 0.001:
        return f"{seconds * 1000000:.0f}µs"
    elif seconds < 1:
        return f"{seconds * 1000:.1f}ms"
    else:
        return f"{seconds:.2f}s"


def format_stats(stats, indent=2):
    indent_str = " " * indent
    lines = []
    for k, v in stats.items():
        if isinstance(v, dict):
            lines.append(f"{indent_str}{k}:")
            lines.append(format_stats(v, indent + 2))
        elif isinstance(v, float):
            lines.append(f"{indent_str}{k}: {v:.4f}")
        else:
            lines.append(f"{indent_str}{k}: {v}")
    return "\n".join(lines)


def run_demo():
    print_section("OTTOMINER PIPELINE DEMO")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    base_dir = Path("/tmp/ottominer_demo")
    pdf_dir = base_dir / "pdfs"
    output_dir = base_dir / "output"

    timings = {}

    print_section("1. CREATING SAMPLE PDFs")
    with Timer("pdf_creation") as t:
        pdf_paths = create_sample_pdfs(pdf_dir, count=3)
    timings["pdf_creation"] = t.elapsed

    print(f"Created {len(pdf_paths)} PDFs in {format_time(t.elapsed)}")
    for p in pdf_paths:
        print(f"  - {p.name}")

    print_section("2. EXTRACTION PHASE")
    from ottominer.extractors.pdf import PDFExtractor

    extractor = PDFExtractor(
        config={
            "pdf_extraction": {
                "dpi": 200,
            }
        }
    )

    extracted = {}
    for pdf_path in pdf_paths:
        with Timer(f"extract_{pdf_path.stem}") as t:
            text = extractor.extract(pdf_path)
        timings[f"extract_{pdf_path.stem}"] = t.elapsed

        extracted[pdf_path.stem] = text
        print(f"\n{pdf_path.name}:")
        print(f"  Time: {format_time(t.elapsed)}")
        print(f"  Chars: {len(text)}")
        print(f"  Preview: {text[:100]}...")

    print_section("3. TOKENIZATION PHASE")
    from ottominer.analyzers.tokenizer import OttomanTokenizer

    tokenizer = OttomanTokenizer()
    tokenized = {}

    for stem, text in extracted.items():
        with Timer(f"tokenize_{stem}") as t:
            tok_doc = tokenizer.tokenize(stem + ".pdf", text)
        timings[f"tokenize_{stem}"] = t.elapsed
        tokenized[stem] = tok_doc

        print(f"\n{stem}:")
        print(f"  Time: {format_time(t.elapsed)}")
        print(f"  Tokens: {len(tok_doc.tokens)}")
        print(f"  Filtered: {len(tok_doc.filtered_tokens)}")
        print(f"  Unique: {len(set(tok_doc.filtered_tokens))}")

    print_section("4. ANALYSIS PHASE")
    from ottominer.analyzers.semantic import SemanticAnalyzer
    from ottominer.analyzers.morphology import MorphologyAnalyzer
    from ottominer.analyzers.genre import GenreAnalyzer
    from ottominer.analyzers.base import AnalyzedDocument

    semantic = SemanticAnalyzer()
    morphology = MorphologyAnalyzer()
    genre = GenreAnalyzer()

    analyzed = {}

    for stem, tok_doc in tokenized.items():
        doc = AnalyzedDocument(tokenized=tok_doc)

        print(f"\n{stem}:")

        with Timer(f"semantic_{stem}") as t:
            semantic.analyze(doc)
        timings[f"semantic_{stem}"] = t.elapsed
        print(
            f"  Semantic: {format_time(t.elapsed)} - {len(doc.semantic_labels)} labeled tokens"
        )

        with Timer(f"morphology_{stem}") as t:
            morphology.analyze(doc)
        timings[f"morphology_{stem}"] = t.elapsed
        print(
            f"  Morphology: {format_time(t.elapsed)} - {len(doc.morphology)} tokens with suffixes"
        )

        with Timer(f"genre_{stem}") as t:
            genre.analyze(doc)
        timings[f"genre_{stem}"] = t.elapsed
        print(f"  Genre: {format_time(t.elapsed)} - register={doc.register}")

        analyzed[stem] = doc

    print_section("5. SIMILARITY ANALYSIS")
    from ottominer.analyzers.similarity import SimilarityAnalyzer

    similarity = SimilarityAnalyzer(use_embeddings=False)
    docs_list = list(analyzed.values())

    with Timer("similarity") as t:
        similarity.compute_batch(docs_list)
    timings["similarity"] = t.elapsed

    print(f"Computed TF-IDF vectors in {format_time(t.elapsed)}")

    import math

    print("\nPairwise Cosine Similarities:")
    stems = list(analyzed.keys())
    for i, s1 in enumerate(stems):
        for j, s2 in enumerate(stems):
            if i < j:
                v1 = analyzed[s1].similarity_vector
                v2 = analyzed[s2].similarity_vector
                if v1 and v2:
                    dot = sum(a * b for a, b in zip(v1, v2))
                    n1 = math.sqrt(sum(x**2 for x in v1))
                    n2 = math.sqrt(sum(x**2 for x in v2))
                    sim = dot / (n1 * n2) if n1 and n2 else 0
                    print(f"  {s1} <-> {s2}: {sim:.3f}")

    print_section("6. AGGREGATE STATISTICS")

    all_tokens = []
    all_semantic = Counter()
    all_morphology = Counter()

    for stem, doc in analyzed.items():
        all_tokens.extend(doc.tokenized.filtered_tokens)
        for labels in doc.semantic_labels.values():
            for l in labels:
                all_semantic[l] += 1
        for types in doc.morphology.values():
            for t in types:
                all_morphology[t] += 1

    print(f"\nCorpus Statistics:")
    print(f"  Total documents: {len(analyzed)}")
    print(f"  Total tokens: {len(all_tokens)}")
    print(f"  Unique tokens: {len(set(all_tokens))}")
    print(f"  Type-token ratio: {len(set(all_tokens)) / len(all_tokens):.3f}")

    print(f"\nTop 10 Tokens:")
    for token, count in Counter(all_tokens).most_common(10):
        print(f"  {token}: {count}")

    print(f"\nSemantic Categories:")
    for cat, count in all_semantic.most_common():
        print(f"  {cat}: {count}")

    print(f"\nMorphological Suffixes:")
    for suffix, count in all_morphology.most_common():
        print(f"  {suffix}: {count}")

    print_section("7. VISUALIZATION")
    from ottominer.visualizers.html_report import generate_html_report

    output_dir.mkdir(parents=True, exist_ok=True)

    for stem, doc in analyzed.items():
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
        json_path = output_dir / f"{stem}.json"
        json_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"  Saved: {json_path.name}")

    with Timer("html_report") as t:
        report_path = generate_html_report(
            [
                json.loads((output_dir / f"{s}.json").read_text())
                for s in analyzed.keys()
            ],
            output_dir,
        )
    timings["html_report"] = t.elapsed

    print(f"\nHTML Report: {report_path}")
    print(f"Generated in {format_time(t.elapsed)}")

    print_section("8. TIMING SUMMARY")

    categories = {
        "PDF Creation": ["pdf_creation"],
        "Extraction": [k for k in timings if k.startswith("extract_")],
        "Tokenization": [k for k in timings if k.startswith("tokenize_")],
        "Semantic Analysis": [k for k in timings if k.startswith("semantic_")],
        "Morphology Analysis": [k for k in timings if k.startswith("morphology_")],
        "Genre Analysis": [k for k in timings if k.startswith("genre_")],
        "Similarity": ["similarity"],
        "Visualization": ["html_report"],
    }

    grand_total = 0
    print(f"\n{'Category':<25} {'Time':>12} {'%':>8}")
    print("-" * 50)

    for cat, keys in categories.items():
        cat_time = sum(timings.get(k, 0) for k in keys)
        grand_total += cat_time
        if cat_time > 0:
            print(f"{cat:<25} {format_time(cat_time):>12}")

    print("-" * 50)
    print(f"{'TOTAL':<25} {format_time(grand_total):>12}")

    print_section("9. OUTPUT FILES")
    print(f"\nOutput directory: {output_dir}")
    for f in sorted(output_dir.iterdir()):
        size = f.stat().st_size
        if size > 1024 * 1024:
            size_str = f"{size / (1024 * 1024):.1f} MB"
        elif size > 1024:
            size_str = f"{size / 1024:.1f} KB"
        else:
            size_str = f"{size} B"
        print(f"  {f.name}: {size_str}")

    print_section("DEMO COMPLETE")
    print(f"\nTo view the HTML report:")
    print(f"  xdg-open {report_path}")
    print(f"\nOr in a browser:")
    print(f"  file://{report_path}")

    return analyzed, timings


if __name__ == "__main__":
    run_demo()
