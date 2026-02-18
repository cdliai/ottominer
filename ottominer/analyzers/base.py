from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional


@dataclass
class TokenizedDocument:
    source_path: str
    raw_text: str
    tokens: List[str]
    lemmas: List[str]
    offsets: List[Tuple[int, int]]
    filtered_tokens: List[str]


@dataclass
class AnalyzedDocument:
    tokenized: TokenizedDocument
    semantic_labels: Dict[str, List[str]] = field(default_factory=dict)
    morphology: Dict[str, List[str]] = field(default_factory=dict)
    genre_scores: Dict[str, float] = field(default_factory=dict)
    register: str = "unknown"
    similarity_vector: Optional[List[float]] = None
