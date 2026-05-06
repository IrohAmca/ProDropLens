"""
Visualization helpers for ProDropLens.

Notebooks call these functions instead of carrying raw plotting code.
"""

from typing import Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Color palette

COLORS = {
    "overt": "#4C9BE8",     # blue, overt subject
    "prodrop": "#E8774C",   # orange, pro-drop
    "positive": "#2ECC71",  # green, positive contribution
    "negative": "#E74C3C",  # red, negative contribution
    "neutral": "#95A5A6",   # gray, neutral
}

PLOTLY_TEMPLATE = "plotly_dark"


# Phase 0 tokenization visualizations

def plot_token_table(
    token_strs: list[str],
    title: str = "",
    highlight_positions: Optional[list[int]] = None,
) -> go.Figure:
    """
    Show a sentence's tokens as colored table cells.

    Args:
        token_strs: token string list
        title: plot title
        highlight_positions: token positions to highlight, such as subject/verb
    """
    highlight_positions = highlight_positions or []
    colors = [
        "#E8774C" if i in highlight_positions else "#2C3E50"
        for i in range(len(token_strs))
    ]
    text_colors = ["white"] * len(token_strs)

    fig = go.Figure(go.Table(
        header=dict(
            values=[f"<b>Pos {i}</b>" for i in range(len(token_strs))],
            fill_color="#1a1a2e",
            font=dict(color="white", size=12),
            align="center",
        ),
        cells=dict(
            values=[[f'"{t}"'] for t in token_strs],
            fill_color=[[color] for color in colors],
            font=dict(color=text_colors, size=13),
            align="center",
            height=40,
        ),
    ))
    fig.update_layout(
        title=title,
        template=PLOTLY_TEMPLATE,
        margin=dict(l=20, r=20, t=50, b=20),
        height=150,
    )
    return fig


def plot_tokenization_comparison(
    overt_info: dict,
    prodrop_info: dict,
    pair_id: str = "",
) -> go.Figure:
    """
    Show overt-subject and pro-drop tokenizations side by side.

    Subject and verb positions are highlighted.
    """
    def make_row(info: dict, label: str) -> dict:
        toks = info["token_strs"]
        subj_pos = info.get("subject_pos")
        verb_pos = info.get("verb_pos")
        colors = []
        for i in range(len(toks)):
            if i == subj_pos:
                colors.append("#4C9BE8")
            elif i == verb_pos:
                colors.append("#E8774C")
            else:
                colors.append("#2C3E50")
        return {"toks": toks, "colors": colors, "label": label}

    rows = [
        make_row(overt_info, "Overt Subject"),
        make_row(prodrop_info, "Pro-Drop"),
    ]

    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=[r["label"] for r in rows],
        vertical_spacing=0.1,
        specs=[[{"type": "table"}], [{"type": "table"}]],
    )

    max_len = max(len(r["toks"]) for r in rows)
    for row_idx, row in enumerate(rows, 1):
        toks = row["toks"] + [""] * (max_len - len(row["toks"]))
        clrs = row["colors"] + ["rgba(0,0,0,0)"] * (max_len - len(row["colors"]))
        fig.add_trace(go.Table(
            header=dict(
                values=[f"Pos {i}" for i in range(max_len)],
                fill_color="#1a1a2e",
                font=dict(color="white", size=11),
            ),
            cells=dict(
                values=[[f'"{t}"' if t else ""] for t in toks],
                fill_color=[[color] for color in clrs],
                font=dict(color="white", size=12),
                align="center",
                height=38,
            ),
        ), row=row_idx, col=1)

    fig.update_layout(
        title=f"Tokenization Comparison - {pair_id}",
        template=PLOTLY_TEMPLATE,
        height=350,
        showlegend=False,
    )
    return fig


def plot_token_id_matrix(token_id_table: pd.DataFrame) -> go.Figure:
    """
    Visualize the summary table of verb forms and token IDs.

    Args:
        token_id_table: columns ["form", "token_id", "n_tokens", "single_token"]
    """
    cell_colors = [
        ["#2ECC71" if v else "#E74C3C" for v in token_id_table["single_token"]]
    ]

    fig = go.Figure(go.Table(
        header=dict(
            values=["<b>Form</b>", "<b>Token ID</b>", "<b>Token Count</b>", "<b>Single Token?</b>"],
            fill_color="#1a1a2e",
            font=dict(color="white", size=13),
            align="center",
        ),
        cells=dict(
            values=[
                token_id_table["form"],
                token_id_table["token_id"].fillna("-"),
                token_id_table["n_tokens"],
                token_id_table["single_token"].map({True: "Yes", False: "No"}),
            ],
            fill_color=["#16213e", "#16213e", "#16213e"] + cell_colors,
            font=dict(color="white", size=12),
            align="center",
            height=35,
        ),
    ))
    fig.update_layout(
        title="Verb Forms - Token ID Table",
        template=PLOTLY_TEMPLATE,
        height=400,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def plot_position_offset_summary(offsets: dict[str, Optional[int]]) -> go.Figure:
    """
    Show the verb-position offset for each minimal pair.

    Offset = overt verb position - pro-drop verb position.

    Args:
        offsets: {"pilot_gitmek_simdiki_1s": 1, ...}
    """
    ids = list(offsets.keys())
    values = [v if v is not None else 0 for v in offsets.values()]
    bar_colors = [COLORS["positive"] if v > 0 else COLORS["neutral"] for v in values]

    fig = go.Figure(go.Bar(
        x=ids,
        y=values,
        marker_color=bar_colors,
        text=values,
        textposition="outside",
    ))
    fig.update_layout(
        title="Verb Position Offset (Overt - Pro-Drop)",
        xaxis_title="Pair ID",
        yaxis_title="Token Offset",
        template=PLOTLY_TEMPLATE,
        height=400,
        xaxis_tickangle=-45,
    )
    return fig
