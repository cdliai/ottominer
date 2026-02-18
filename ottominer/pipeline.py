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
        self._morphology = (
            MorphologyAnalyzer() if "morphology" in self.analyzers else None
        )
        self._genre = GenreAnalyzer() if "genre" in self.analyzers else None
        self._similarity = (
            SimilarityAnalyzer(use_embeddings=use_embeddings)
            if "similarity" in self.analyzers
            else None
        )

    def run(
        self, pdf_paths: List[Union[str, Path]]
    ) -> List[Optional[AnalyzedDocument]]:
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

        if self._similarity and analyzed:
            valid = [d for d in analyzed if d is not None]
            if valid:
                self._similarity.compute_batch(valid)

        for doc in analyzed:
            if doc is not None:
                self._save_json(doc)

        return analyzed

    def _process_one(self, path: Path) -> Optional[AnalyzedDocument]:
        """Extract, tokenize, and analyze one document. Returns None on failure."""
        try:
            raw_text = self._extractor.extract(path)
            if not raw_text:
                logger.warning(f"Empty extraction result for {path}")
                raw_text = ""
        except Exception as e:
            logger.error(f"Extraction failed for {path}: {e}")
            return None

        try:
            tokenized: TokenizedDocument = self._tokenizer.tokenize(
                source_path=str(path),
                text=raw_text,
            )
        except Exception as e:
            logger.error(f"Tokenization failed for {path}: {e}")
            return None

        doc = AnalyzedDocument(tokenized=tokenized)

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
            out_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            logger.info(f"Saved analysis to {out_path}")
        except Exception as e:
            logger.error(f"Failed to save JSON for {doc.tokenized.source_path}: {e}")

        return out_path

    def _process_one_direct(self, doc: AnalyzedDocument) -> AnalyzedDocument:
        """Run analyzers on an already-tokenized document (for CLI analyze command)."""
        try:
            if self._semantic:
                self._semantic.analyze(doc)
            if self._morphology:
                self._morphology.analyze(doc)
            if self._genre:
                self._genre.analyze(doc)
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
        return doc
