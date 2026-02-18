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
