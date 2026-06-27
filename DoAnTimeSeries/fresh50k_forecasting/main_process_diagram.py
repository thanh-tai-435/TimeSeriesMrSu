"""Generate process diagram assets for the Vietnamese report."""

from __future__ import annotations

from html import escape
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from src.config import FIGURES_DIR


NODES = [
    ("raw", "Dữ liệu gốc\ntrain/eval parquet", 0, 8.8, "#ffe6cc", "#d79b00"),
    ("hourly", "Bung hourly\nhours_sale + stockout", 0, 7.5, "#dae8fc", "#6c8ebf"),
    ("features", "Feature engineering\nlag/rolling/calendar", 0, 6.2, "#dae8fc", "#6c8ebf"),
    ("split", "Time split\nTrain / Val / Test", 0, 4.9, "#fff2cc", "#d6b656"),
    ("warmup", "Warmup 14 ngày\nGiữ observed sales", -3.0, 3.4, "#f5f5f5", "#666666"),
    ("blocks", "Train period\nExpanding weekly blocks", 0, 3.4, "#e1d5e7", "#9673a6"),
    ("final", "Final recovery model\nTrain-period non-stockout", 3.0, 3.4, "#e1d5e7", "#9673a6"),
    ("recover", "Recovered demand\nmax(observed, imputed)", 0, 2.0, "#d5e8d4", "#82b366"),
    ("daily", "Aggregate daily\n7-day demand target", 0, 0.7, "#d5e8d4", "#82b366"),
    ("forecast", "Forecasting\nObserved vs Recovered", 0, -0.6, "#dae8fc", "#6c8ebf"),
    ("eval", "Evaluation\nWAPE, WPE, bias", 0, -1.9, "#fff2cc", "#d6b656"),
]

EDGES = [
    ("raw", "hourly"),
    ("hourly", "features"),
    ("features", "split"),
    ("split", "warmup"),
    ("split", "blocks"),
    ("split", "final"),
    ("warmup", "recover"),
    ("blocks", "recover"),
    ("final", "recover"),
    ("recover", "daily"),
    ("daily", "forecast"),
    ("forecast", "eval"),
]

RECOVERY_NODES = [
    ("features", "Hourly features\nsale + stockout + lags", 0, 9.0, "#dae8fc", "#6c8ebf"),
    ("split", "Time split\ntrain / val / test", 0, 7.8, "#fff2cc", "#d6b656"),
    ("warmup", "Warmup 14 days\nkeep observed sales", 0, 6.6, "#f5f5f5", "#666666"),
    ("block", "Current train block k\n7-day window", 0, 5.4, "#e1d5e7", "#9673a6"),
    ("pool", "Training pool\npast non-stockout rows only", 0, 4.2, "#d5e8d4", "#82b366"),
    ("fit", "Fit recovery model\nLightGBM", 0, 3.0, "#dae8fc", "#6c8ebf"),
    ("predict", "Predict block k\nimputed demand", 0, 1.8, "#dae8fc", "#6c8ebf"),
    ("recover", "Apply stockout rule\nmax(observed, imputed)", 0, 0.6, "#d5e8d4", "#82b366"),
    ("advance", "Advance next block\nnon-stockout rows\nenter future pool", 3.4, 1.8, "#fff2cc", "#d6b656"),
    ("final", "Final recovery model\ntrain non-stockout\nrows only", 0, -0.8, "#e1d5e7", "#9673a6"),
    ("forward", "Recover validation/test\nno fit on validation/test", 0, -2.0, "#f8cecc", "#b85450"),
    ("daily", "Aggregate recovered hourly\nto daily latent demand", 0, -3.2, "#d5e8d4", "#82b366"),
]

RECOVERY_EDGES = [
    ("features", "split"),
    ("split", "warmup"),
    ("warmup", "block"),
    ("block", "pool"),
    ("pool", "fit"),
    ("fit", "predict"),
    ("predict", "recover"),
    ("recover", "advance"),
    ("advance", "block"),
    ("recover", "final"),
    ("final", "forward"),
    ("forward", "daily"),
]


def _node_dict(nodes):
    return {node[0]: node for node in nodes}


def draw_png(output_path: Path) -> None:
    """Draw a clean PNG flowchart for LaTeX."""
    nodes = _node_dict(NODES)
    fig, ax = plt.subplots(figsize=(10, 12))
    ax.set_xlim(-5, 5)
    ax.set_ylim(-2.8, 9.6)
    ax.axis("off")

    for _, label, x, y, fill, stroke in NODES:
        box = FancyBboxPatch(
            (x - 1.25, y - 0.38),
            2.5,
            0.76,
            boxstyle="round,pad=0.03,rounding_size=0.06",
            linewidth=1.6,
            edgecolor=stroke,
            facecolor=fill,
        )
        ax.add_patch(box)
        ax.text(x, y, label, ha="center", va="center", fontsize=10)

    for source, target in EDGES:
        _, _, sx, sy, _, _ = nodes[source]
        _, _, tx, ty, _, _ = nodes[target]
        arrow = FancyArrowPatch(
            (sx, sy - 0.42),
            (tx, ty + 0.42),
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.2,
            color="#444444",
            connectionstyle="arc3,rad=0.0",
        )
        ax.add_patch(arrow)

    ax.text(
        0,
        9.35,
        "Quy trình stockout-aware latent demand recovery và forecasting",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
    )
    ax.text(
        0,
        -2.45,
        "Nguyên tắc: chỉ dùng dữ liệu quá khứ để recover block hiện tại; validation/test không được dùng để fit recovery model.",
        ha="center",
        va="center",
        fontsize=9,
        color="#444444",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def draw_recovery_png(output_path: Path) -> None:
    """Draw a focused expanding-window recovery flowchart for LaTeX."""
    nodes = _node_dict(RECOVERY_NODES)
    fig, ax = plt.subplots(figsize=(9, 13))
    ax.set_xlim(-4.6, 5.0)
    ax.set_ylim(-4.3, 9.8)
    ax.axis("off")

    for _, label, x, y, fill, stroke in RECOVERY_NODES:
        box = FancyBboxPatch(
            (x - 1.35, y - 0.42),
            2.7,
            0.84,
            boxstyle="round,pad=0.03,rounding_size=0.06",
            linewidth=1.6,
            edgecolor=stroke,
            facecolor=fill,
        )
        ax.add_patch(box)
        ax.text(x, y, label, ha="center", va="center", fontsize=9.5)

    for source, target in RECOVERY_EDGES:
        _, _, sx, sy, _, _ = nodes[source]
        _, _, tx, ty, _, _ = nodes[target]
        start = (sx, sy - 0.47)
        end = (tx, ty + 0.47)
        rad = 0.0
        if source == "advance" and target == "block":
            start = (sx - 1.35, sy + 0.05)
            end = (tx + 1.35, ty - 0.05)
            rad = 0.35
        elif source == "recover" and target == "advance":
            start = (sx + 1.35, sy)
            end = (tx - 1.35, ty)
            rad = 0.0
        arrow = FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.2,
            color="#444444",
            connectionstyle=f"arc3,rad={rad}",
        )
        ax.add_patch(arrow)

    ax.text(
        0,
        9.55,
        "Quy trình xử lý riêng cho expanding-window recovery",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
    )
    ax.text(
        0,
        -4.0,
        "Mỗi block chỉ nhìn về quá khứ: không dùng ngày phía sau để recover ngày phía trước; validation/test không được dùng để fit recovery model.",
        ha="center",
        va="center",
        fontsize=9,
        color="#444444",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_drawio(output_path: Path, nodes=None, edges=None, name: str = "Process") -> None:
    """Write an editable draw.io XML source file."""
    nodes = NODES if nodes is None else nodes
    edges = EDGES if edges is None else edges
    style_tpl = (
        "rounded=1;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};"
        "fontSize=12;fontFamily=Arial;"
    )
    cells = [
        '<mxCell id="0" />',
        '<mxCell id="1" parent="0" />',
    ]
    positions = {}
    y_values = [node[3] for node in nodes]
    max_y = max(y_values) + 0.8
    for idx, (node_id, label, x, y, fill, stroke) in enumerate(nodes, start=2):
        mx_x = int((x + 5) * 90)
        mx_y = int((max_y - y) * 80)
        positions[node_id] = str(idx)
        value = escape(label).replace("\n", "&#xa;")
        style = style_tpl.format(fill=fill, stroke=stroke)
        cells.append(
            f'<mxCell id="{idx}" value="{value}" style="{style}" vertex="1" parent="1">'
            f'<mxGeometry x="{mx_x}" y="{mx_y}" width="210" height="70" as="geometry" />'
            "</mxCell>"
    )
    edge_id = 100
    for source, target in edges:
        cells.append(
            f'<mxCell id="{edge_id}" value="" style="edgeStyle=orthogonalEdgeStyle;rounded=1;'
            f'orthogonalLoop=1;jettySize=auto;html=1;endArrow=block;" edge="1" parent="1" '
            f'source="{positions[source]}" target="{positions[target]}">'
            '<mxGeometry relative="1" as="geometry" />'
            "</mxCell>"
        )
        edge_id += 1
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<mxfile host="drawio" version="26.0.0">\n'
        f'  <diagram name="{escape(name)}">\n'
        '    <mxGraphModel dx="1200" dy="900" grid="1" gridSize="10" guides="1" tooltips="1" '
        'connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="1000" pageHeight="1400" '
        'math="0" shadow="0">\n'
        "      <root>\n"
        + "\n".join(f"        {cell}" for cell in cells)
        + "\n      </root>\n"
        "    </mxGraphModel>\n"
        "  </diagram>\n"
        "</mxfile>\n"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(xml, encoding="utf-8")


def main() -> None:
    draw_png(FIGURES_DIR / "owner_expanding_window_process.png")
    draw_recovery_png(FIGURES_DIR / "owner_expanding_window_recovery_detail.png")
    write_drawio(FIGURES_DIR / "owner_expanding_window_process.drawio")
    write_drawio(
        FIGURES_DIR / "owner_expanding_window_recovery_detail.drawio",
        nodes=RECOVERY_NODES,
        edges=RECOVERY_EDGES,
        name="Expanding Window Recovery Detail",
    )
    print(f"Saved diagram PNG: {FIGURES_DIR / 'owner_expanding_window_process.png'}")
    print(f"Saved recovery detail PNG: {FIGURES_DIR / 'owner_expanding_window_recovery_detail.png'}")
    print(f"Saved diagram source: {FIGURES_DIR / 'owner_expanding_window_process.drawio'}")
    print(f"Saved recovery detail source: {FIGURES_DIR / 'owner_expanding_window_recovery_detail.drawio'}")


if __name__ == "__main__":
    main()
