import logging
from pathlib import Path
from typing import List, Dict, Any
from collections import Counter
import math

logger = logging.getLogger(__name__)


def generate_figures(
    docs: List[Dict[str, Any]], output_dir: Path, dpi: int = 300
) -> List[Path]:
    """Generate static publication-ready figures using matplotlib."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.error("matplotlib not installed. Install with: pip install matplotlib")
        return []

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid")

    generated = []

    fig = _create_token_frequency_figure(docs, dpi)
    if fig:
        for ext in ["png", "pdf"]:
            path = output_dir / f"token_frequency.{ext}"
            fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
            generated.append(path)
        plt.close(fig)

    fig = _create_semantic_distribution_figure(docs, dpi)
    if fig:
        for ext in ["png", "pdf"]:
            path = output_dir / f"semantic_distribution.{ext}"
            fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
            generated.append(path)
        plt.close(fig)

    fig = _create_entropy_figure(docs, dpi)
    if fig:
        for ext in ["png", "pdf"]:
            path = output_dir / f"entropy_analysis.{ext}"
            fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
            generated.append(path)
        plt.close(fig)

    fig = _create_similarity_matrix_figure(docs, dpi)
    if fig:
        for ext in ["png", "pdf"]:
            path = output_dir / f"similarity_matrix.{ext}"
            fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
            generated.append(path)
        plt.close(fig)

    fig = _create_morphology_figure(docs, dpi)
    if fig:
        for ext in ["png", "pdf"]:
            path = output_dir / f"morphology_breakdown.{ext}"
            fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
            generated.append(path)
        plt.close(fig)

    logger.info(f"Generated {len(generated)} figure files")
    return generated


def _create_token_frequency_figure(docs: List[Dict], dpi: int) -> Any:
    """Create token frequency bar chart."""
    import matplotlib.pyplot as plt

    all_tokens = []
    for doc in docs:
        all_tokens.extend(doc.get("filtered_tokens", []))

    if not all_tokens:
        return None

    counter = Counter(all_tokens)
    top_20 = counter.most_common(20)

    fig, ax = plt.subplots(figsize=(10, 6))

    tokens = [t[0] for t in top_20]
    freqs = [t[1] for t in top_20]

    bars = ax.barh(range(len(tokens)), freqs, color="#2E86AB")
    ax.set_yticks(range(len(tokens)))
    ax.set_yticklabels(tokens)
    ax.invert_yaxis()
    ax.set_xlabel("Frequency")
    ax.set_title("Top 20 Token Frequencies")

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="both", length=0)

    plt.tight_layout()
    return fig


def _create_semantic_distribution_figure(docs: List[Dict], dpi: int) -> Any:
    """Create semantic category distribution pie chart."""
    import matplotlib.pyplot as plt
    from collections import defaultdict

    category_counts = defaultdict(int)
    for doc in docs:
        for token, labels in doc.get("semantic_labels", {}).items():
            for label in labels:
                category_counts[label] += 1

    if not category_counts:
        return None

    fig, ax = plt.subplots(figsize=(8, 8))

    labels = list(category_counts.keys())
    sizes = list(category_counts.values())
    colors = _get_mpl_colors(len(labels))

    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        autopct="%1.1f%%",
        colors=colors,
        startangle=90,
        pctdistance=0.8,
    )

    for text in texts:
        text.set_fontsize(10)
    for autotext in autotexts:
        autotext.set_fontsize(9)
        autotext.set_color("white")

    ax.set_title("Semantic Category Distribution")

    plt.tight_layout()
    return fig


def _create_entropy_figure(docs: List[Dict], dpi: int) -> Any:
    """Create entropy analysis line chart."""
    import matplotlib.pyplot as plt

    entropies = []
    doc_names = []

    for i, doc in enumerate(docs):
        tokens = doc.get("filtered_tokens", [])
        if not tokens:
            continue

        counter = Counter(tokens)
        total = len(tokens)
        entropy = -sum(
            (count / total) * math.log2(count / total) for count in counter.values()
        )

        entropies.append(entropy)
        doc_names.append(Path(doc.get("source_path", f"doc_{i}")).stem)

    if not entropies:
        return None

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(range(len(entropies)), entropies, marker="o", color="#2E86AB", linewidth=2)
    ax.fill_between(range(len(entropies)), entropies, alpha=0.3, color="#2E86AB")

    ax.set_xlabel("Document")
    ax.set_ylabel("Entropy (bits)")
    ax.set_title("Token Entropy per Document")

    if len(doc_names) <= 20:
        ax.set_xticks(range(len(doc_names)))
        ax.set_xticklabels(doc_names, rotation=45, ha="right", fontsize=8)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="both", length=0)

    plt.tight_layout()
    return fig


def _create_similarity_matrix_figure(docs: List[Dict], dpi: int) -> Any:
    """Create similarity matrix heatmap."""
    import matplotlib.pyplot as plt
    import numpy as np

    vectors = []
    doc_names = []

    for doc in docs:
        vec = doc.get("similarity_vector")
        if vec:
            vectors.append(vec)
            doc_names.append(Path(doc.get("source_path", f"doc_{len(doc_names)}")).stem)

    if len(vectors) < 2:
        return None

    similarity_matrix = []
    for i, v1 in enumerate(vectors):
        row = []
        for j, v2 in enumerate(vectors):
            if i == j:
                row.append(1.0)
            else:
                dot = sum(a * b for a, b in zip(v1, v2))
                norm1 = math.sqrt(sum(x**2 for x in v1))
                norm2 = math.sqrt(sum(x**2 for x in v2))
                if norm1 and norm2:
                    row.append(dot / (norm1 * norm2))
                else:
                    row.append(0.0)
        similarity_matrix.append(row)

    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(similarity_matrix, cmap="RdBu", vmin=0, vmax=1)

    ax.set_xticks(range(len(doc_names)))
    ax.set_yticks(range(len(doc_names)))
    ax.set_xticklabels(doc_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(doc_names, fontsize=8)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Cosine Similarity")

    ax.set_title("Document Similarity Matrix")

    plt.tight_layout()
    return fig


def _create_morphology_figure(docs: List[Dict], dpi: int) -> Any:
    """Create morphology breakdown bar chart."""
    import matplotlib.pyplot as plt
    from collections import defaultdict

    suffix_totals = defaultdict(int)
    for doc in docs:
        for token, types in doc.get("morphology", {}).items():
            for t in types:
                suffix_totals[t] += 1

    if not suffix_totals:
        return None

    sorted_suffixes = sorted(suffix_totals.items(), key=lambda x: x[1], reverse=True)

    if len(sorted_suffixes) > 15:
        sorted_suffixes = sorted_suffixes[:15]

    fig, ax = plt.subplots(figsize=(10, 6))

    suffix_types = [s[0] for s in sorted_suffixes]
    counts = [s[1] for s in sorted_suffixes]

    bars = ax.barh(range(len(suffix_types)), counts, color="#A23B72")
    ax.set_yticks(range(len(suffix_types)))
    ax.set_yticklabels(suffix_types)
    ax.invert_yaxis()
    ax.set_xlabel("Count")
    ax.set_title("Suffix Type Distribution")

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="both", length=0)

    plt.tight_layout()
    return fig


def _get_mpl_colors(n: int) -> List[str]:
    """Get a colorblind-friendly color palette for matplotlib."""
    base_colors = [
        "#2E86AB",
        "#A23B72",
        "#F18F01",
        "#3B1F2B",
        "#95C623",
        "#1B998B",
        "#ED217C",
        "#7B68EE",
        "#FF6B6B",
        "#4ECDC4",
        "#45B7D1",
        "#96CEB4",
        "#FFEAA7",
        "#DDA0DD",
        "#C73E1D",
    ]
    return (base_colors * ((n // len(base_colors)) + 1))[:n]
