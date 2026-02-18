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
