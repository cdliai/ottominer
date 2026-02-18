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
