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
            source_path="x.pdf",
            raw_text="",
            tokens=[],
            lemmas=[],
            offsets=[],
            filtered_tokens=[],
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

        sa = SimilarityAnalyzer()
        docs = self._make_docs(["namaz ibadet din", "namaz ibadet din"])
        result = sa.compute_batch(docs)
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
