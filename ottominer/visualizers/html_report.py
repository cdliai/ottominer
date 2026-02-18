from __future__ import annotations
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import Counter
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    import plotly.graph_objects as go

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    go = None


def generate_html_report(docs: List[Dict[str, Any]], output_dir: Path) -> Path:
    """Generate an interactive HTML report using Plotly."""
    if not PLOTLY_AVAILABLE:
        logger.error("plotly not installed. Install with: pip install plotly")
        return _generate_fallback_html(docs, output_dir)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.html"

    figures_html = []

    token_freq_chart = _create_token_frequency_chart(docs)
    if token_freq_chart:
        figures_html.append(
            token_freq_chart.to_html(full_html=False, include_plotlyjs=False)
        )

    semantic_chart = _create_semantic_distribution_chart(docs)
    if semantic_chart:
        figures_html.append(
            semantic_chart.to_html(full_html=False, include_plotlyjs=False)
        )

    register_chart = _create_register_chart(docs)
    if register_chart:
        figures_html.append(
            register_chart.to_html(full_html=False, include_plotlyjs=False)
        )

    morphology_chart = _create_morphology_heatmap(docs)
    if morphology_chart:
        figures_html.append(
            morphology_chart.to_html(full_html=False, include_plotlyjs=False)
        )

    similarity_chart = _create_similarity_heatmap(docs)
    if similarity_chart:
        figures_html.append(
            similarity_chart.to_html(full_html=False, include_plotlyjs=False)
        )

    html_content = _wrap_html(figures_html, docs)

    report_path.write_text(html_content, encoding="utf-8")
    logger.info(f"Generated HTML report: {report_path}")

    return report_path


def _create_token_frequency_chart(docs: List[Dict]) -> Optional[Any]:
    """Create a bar chart of top token frequencies."""
    if not PLOTLY_AVAILABLE:
        return None

    all_tokens = []
    for doc in docs:
        all_tokens.extend(doc.get("filtered_tokens", []))

    if not all_tokens:
        return None

    counter = Counter(all_tokens)
    top_30 = counter.most_common(30)

    fig = go.Figure(
        [
            go.Bar(
                x=[t[0] for t in top_30],
                y=[t[1] for t in top_30],
                marker_color="#2E86AB",
            )
        ]
    )

    fig.update_layout(
        title="Top 30 Token Frequencies",
        xaxis_title="Token",
        yaxis_title="Frequency",
        xaxis_tickangle=-45,
        height=400,
        margin=dict(b=100),
    )

    return fig


def _create_semantic_distribution_chart(docs: List[Dict]) -> Optional[Any]:
    """Create a donut chart of semantic category distribution."""
    if not PLOTLY_AVAILABLE:
        return None

    from collections import defaultdict

    category_counts = defaultdict(int)
    for doc in docs:
        for token, labels in doc.get("semantic_labels", {}).items():
            for label in labels:
                category_counts[label] += 1

    if not category_counts:
        return None

    fig = go.Figure(
        [
            go.Pie(
                labels=list(category_counts.keys()),
                values=list(category_counts.values()),
                hole=0.4,
                marker_colors=_get_color_palette(len(category_counts)),
            )
        ]
    )

    fig.update_layout(title="Semantic Category Distribution", height=400)

    return fig


def _create_register_chart(docs: List[Dict]) -> Optional[Any]:
    """Create a horizontal bar chart of register distribution."""
    if not PLOTLY_AVAILABLE:
        return None

    register_counts = {"formal": 0, "informal": 0, "mixed": 0, "unknown": 0}

    for doc in docs:
        register = doc.get("register", "unknown")
        if register in register_counts:
            register_counts[register] += 1

    if all(v == 0 for v in register_counts.values()):
        return None

    fig = go.Figure(
        [
            go.Bar(
                x=list(register_counts.values()),
                y=list(register_counts.keys()),
                orientation="h",
                marker_color=["#2E86AB", "#A23B72", "#F18F01", "#C73E1D"],
            )
        ]
    )

    fig.update_layout(
        title="Document Register Distribution",
        xaxis_title="Document Count",
        yaxis_title="Register",
        height=300,
    )

    return fig


def _create_morphology_heatmap(docs: List[Dict]) -> Optional[Any]:
    """Create a heatmap of suffix type frequencies per document."""
    if not PLOTLY_AVAILABLE:
        return None

    from collections import defaultdict

    if not docs or len(docs) < 2:
        return None

    suffix_types = set()
    doc_suffix_counts = []

    for doc in docs:
        counts = defaultdict(int)
        for token, types in doc.get("morphology", {}).items():
            for t in types:
                counts[t] += 1
                suffix_types.add(t)
        doc_suffix_counts.append(counts)

    if not suffix_types:
        return None

    suffix_types = sorted(suffix_types)
    doc_names = [
        Path(doc.get("source_path", f"doc_{i}")).stem for i, doc in enumerate(docs)
    ]

    z_data = []
    for counts in doc_suffix_counts:
        z_data.append([counts.get(st, 0) for st in suffix_types])

    fig = go.Figure(
        [go.Heatmap(z=z_data, x=suffix_types, y=doc_names, colorscale="Blues")]
    )

    fig.update_layout(
        title="Morphology Analysis (Suffix Types per Document)",
        xaxis_title="Suffix Type",
        yaxis_title="Document",
        height=max(300, len(docs) * 30),
    )

    return fig


def _create_similarity_heatmap(docs: List[Dict]) -> Optional[Any]:
    """Create a heatmap of document similarity matrix."""
    if not PLOTLY_AVAILABLE:
        return None

    import math

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

    fig = go.Figure(
        [
            go.Heatmap(
                z=similarity_matrix,
                x=doc_names,
                y=doc_names,
                colorscale="RdBu",
                zmin=0,
                zmax=1,
            )
        ]
    )

    fig.update_layout(
        title="Document Similarity Matrix (Cosine)",
        height=max(400, len(doc_names) * 40),
    )

    return fig


def _get_color_palette(n: int) -> List[str]:
    """Get a colorblind-friendly color palette."""
    base_colors = [
        "#2E86AB",
        "#A23B72",
        "#F18F01",
        "#C73E1D",
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
    ]
    return (base_colors * ((n // len(base_colors)) + 1))[:n]


def _wrap_html(figures_html: List[str], docs: List[Dict]) -> str:
    """Wrap figure HTML in a complete HTML document."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    figures_section = "\n".join(
        f'<div class="figure-container">{fig}</div>' for fig in figures_html
    )

    stats = _compute_summary_stats(docs)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OttoMiner Analysis Report</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        h1 {{
            color: #2E86AB;
            border-bottom: 2px solid #2E86AB;
            padding-bottom: 10px;
        }}
        .summary {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }}
        .stat-card {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 6px;
            text-align: center;
        }}
        .stat-value {{
            font-size: 2em;
            font-weight: bold;
            color: #2E86AB;
        }}
        .stat-label {{
            color: #666;
            font-size: 0.9em;
        }}
        .figure-container {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .timestamp {{
            color: #666;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <h1>OttoMiner Analysis Report</h1>
    <p class="timestamp">Generated: {timestamp}</p>
    
    <div class="summary">
        <h2>Summary Statistics</h2>
        <div class="summary-grid">
            <div class="stat-card">
                <div class="stat-value">{stats["total_docs"]}</div>
                <div class="stat-label">Documents</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{stats["total_tokens"]:,}</div>
                <div class="stat-label">Total Tokens</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{stats["unique_tokens"]:,}</div>
                <div class="stat-label">Unique Tokens</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{stats["labeled_tokens"]:,}</div>
                <div class="stat-label">Semantically Labeled</div>
            </div>
        </div>
    </div>
    
    {figures_section}
</body>
</html>"""


def _compute_summary_stats(docs: List[Dict]) -> Dict[str, int]:
    """Compute summary statistics for the report."""
    all_tokens = []
    labeled_count = 0

    for doc in docs:
        tokens = doc.get("filtered_tokens", [])
        all_tokens.extend(tokens)
        labeled_count += len(doc.get("semantic_labels", {}))

    return {
        "total_docs": len(docs),
        "total_tokens": len(all_tokens),
        "unique_tokens": len(set(all_tokens)),
        "labeled_tokens": labeled_count,
    }


def _generate_fallback_html(docs: List[Dict], output_dir: Path) -> Path:
    """Generate a simple HTML report without Plotly."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.html"

    stats = _compute_summary_stats(docs)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    doc_rows = []
    for doc in docs:
        source = doc.get("source_path", "unknown")
        register = doc.get("register", "unknown")
        tokens = len(doc.get("filtered_tokens", []))
        doc_rows.append(
            f"<tr><td>{source}</td><td>{register}</td><td>{tokens}</td></tr>"
        )

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>OttoMiner Report</title>
    <style>
        body {{ font-family: sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background: #2E86AB; color: white; }}
    </style>
</head>
<body>
    <h1>OttoMiner Report</h1>
    <p>Generated: {timestamp}</p>
    <p>Documents: {stats["total_docs"]} | Tokens: {stats["total_tokens"]} | Unique: {stats["unique_tokens"]}</p>
    <h2>Documents</h2>
    <table>
        <tr><th>Source</th><th>Register</th><th>Tokens</th></tr>
        {"".join(doc_rows)}
    </table>
</body>
</html>"""

    report_path.write_text(html, encoding="utf-8")
    return report_path
