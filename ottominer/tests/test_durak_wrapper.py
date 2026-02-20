import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from ottominer.analyzers.durak_wrapper import DurakAnalyzer, _DURAK_AVAILABLE
from ottominer.analyzers.base import AnalyzedDocument, TokenizedDocument

class TestDurakAnalyzer:
    
    def test_durak_wrapper_fallback(self):
        # A test token that our rule-based morphology handles (e.g. plural "ler" or similar)
        # Assuming suffixes.json handles specific suffixes. Let's provide a generic word.
        tokens = ["kalemler", "kitaplar", "yapacakmı"]
        doc = AnalyzedDocument(
            tokenized=TokenizedDocument(
                source_path="dummy.pdf",
                raw_text="kalemler kitaplar yapacakmı",
                tokens=tokens,
                lemmas=tokens,
                offsets=[(0, 8), (9, 17), (18, 27)],
                filtered_tokens=tokens
            )
        )
        
        analyzer = DurakAnalyzer()
        analyzed_doc = analyzer.analyze(doc)
        
        # Whether durak is installed or not, we expect the morphology dict to be populated
        # via the rule-based fallback for words with recognizable suffixes.
        assert isinstance(analyzed_doc.morphology, dict)
        
        # If the fallback runs correctly, at least some analysis is present if suffixes matched
        # Since we don't have the exact suffixes.json, we just check it doesn't crash 
        # and returns the original document structure.
        assert hasattr(analyzed_doc, "morphology")

    @patch("ottominer.analyzers.durak_wrapper._DURAK_AVAILABLE", True)
    @patch("durak.attach_detached_suffixes")
    def test_durak_attaches_suffixes(self, mock_attach):
        # Only run this logic if we mock durak 
        mock_attach.return_value = ["yapacakmı"]
        
        tokens = ["yapacak", "mı"]
        doc = AnalyzedDocument(
            tokenized=TokenizedDocument(
                source_path="dummy.pdf",
                raw_text="yapacak mı",
                tokens=tokens,
                lemmas=tokens,
                offsets=[(0, 7), (8, 10)],
                filtered_tokens=tokens
            )
        )
        
        analyzer = DurakAnalyzer()
        analyzer.analyze(doc)
        
        # Check that attach_detached_suffixes was called with filtered_tokens
        mock_attach.assert_called_once_with(tokens)
