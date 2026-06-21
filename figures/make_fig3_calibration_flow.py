from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import patches


# Editable text for SVG/PDF post-processing.
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"]
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42


BLACK = "#111111"
WHITE = "#FFFFFF"
LIGHT = "#F5F5F5"
CANVAS_W = 1500
CANVAS_H = 390


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 8.8,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "savefig.facecolor": "white",
        }
    )


def add_box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    label: str,
    *,
    rounded: bool = False,
    dashed: bool = False,
    lw: float = 1.2,
    fontsize: float = 8.8,
    facecolor: str = WHITE,
) -> None:
    boxstyle = "round,pad=0.015,rounding_size=8" if rounded else "square,pad=0.0"
    ax.add_patch(
        patches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle=boxstyle,
            linewidth=lw,
            edgecolor=BLACK,
            facecolor=facecolor,
            linestyle="--" if dashed else "-",
            zorder=2,
        )
    )
    ax.text(
        x + w / 2,
        y + h / 2,
        label,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=BLACK,
        linespacing=1.12,
        zorder=3,
    )


def add_output(ax, x: float, y: float, w: float, h: float, label: str) -> None:
    ax.add_patch(
        patches.Ellipse(
            (x + w / 2, y + h / 2),
            w,
            h,
            linewidth=1.2,
            edgecolor=BLACK,
            facecolor=WHITE,
            zorder=2,
        )
    )
    ax.text(
        x + w / 2,
        y + h / 2,
        label,
        ha="center",
        va="center",
        fontsize=8.8,
        color=BLACK,
        linespacing=1.12,
        zorder=3,
    )


def arrow(
    ax,
    p0: tuple[float, float],
    p1: tuple[float, float],
    *,
    dashed: bool = False,
    lw: float = 1.15,
    mutation_scale: float = 9.5,
) -> None:
    ax.add_patch(
        patches.FancyArrowPatch(
            p0,
            p1,
            arrowstyle="-|>",
            mutation_scale=mutation_scale,
            linewidth=lw,
            color=BLACK,
            linestyle="--" if dashed else "-",
            shrinkA=0,
            shrinkB=0,
            zorder=1,
        )
    )


def elbow_arrow(
    ax,
    pts: list[tuple[float, float]],
    *,
    dashed: bool = False,
    lw: float = 1.05,
) -> None:
    for p0, p1 in zip(pts[:-2], pts[1:-1]):
        ax.plot(
            [p0[0], p1[0]],
            [p0[1], p1[1]],
            color=BLACK,
            lw=lw,
            linestyle="--" if dashed else "-",
            zorder=1,
        )
    arrow(ax, pts[-2], pts[-1], dashed=dashed, lw=lw)


def add_small_label(ax, x: float, y: float, label: str) -> None:
    ax.text(
        x,
        y,
        label,
        ha="center",
        va="center",
        fontsize=7.8,
        color=BLACK,
        zorder=4,
    )


def build_figure() -> plt.Figure:
    setup_style()
    fig, ax = plt.subplots(1, 1, figsize=(7.2, 2.15))
    ax.set_xlim(0, CANVAS_W)
    ax.set_ylim(CANVAS_H, 0)
    ax.axis("off")

    raw = (35, 155, 155, 72)
    intra = (245, 155, 175, 72)
    streams = (475, 155, 150, 72)
    predict = (680, 155, 150, 72)
    target = (885, 155, 150, 72)
    fit = (1090, 155, 150, 72)
    output = (1295, 155, 155, 72)

    add_box(ax, *raw, "Raw codes\n$D_c$, $F$")
    add_box(ax, *intra, "Intra-group\nalign")
    add_box(ax, *streams, "A/B streams\n$U_A$, $U_B$")
    add_box(ax, *predict, "Predict\n$\\hat{D}_B$")
    add_box(ax, *target, "Fine target\n$\\hat{F}_B$")
    add_box(ax, *fit, "Fit\n$g_B, o_B, k_B$")
    add_output(ax, *output, "Output\n$D_{corr}$")

    mid_y = 191
    for x0, x1 in [(190, 245), (420, 475), (625, 680), (830, 885), (1035, 1090), (1240, 1295)]:
        arrow(ax, (x0, mid_y), (x1, mid_y))

    add_box(ax, 680, 45, 150, 58, "$P\\{U_A\\}$", rounded=True, dashed=True, facecolor=LIGHT)
    elbow_arrow(ax, [(755, 103), (755, 130), (755, 155)], dashed=True)

    add_box(ax, 1085, 270, 160, 58, "Safe\nsamples", rounded=True, dashed=True, facecolor=LIGHT)
    elbow_arrow(ax, [(1165, 270), (1165, 248), (1165, 227)], dashed=True)

    elbow_arrow(ax, [(550, 227), (550, 335), (1372, 335), (1372, 227)], dashed=True)
    add_small_label(ax, 960, 318, "coarse bypass")

    add_small_label(ax, 960, 248, "$\\hat{F}_B=\\hat{D}_B-D_{c,B}$")

    fig.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.04)
    return fig


def save_figure(fig: plt.Figure, base: Path) -> None:
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(
        base.with_suffix(".tiff"),
        dpi=600,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")


def html_label(text: str) -> str:
    return escape(text).replace("\n", "&lt;br&gt;")


def mx_vertex(
    cell_id: str,
    label: str,
    style: str,
    x: int,
    y: int,
    w: int,
    h: int,
) -> str:
    return (
        f'        <mxCell id="{cell_id}" value="{html_label(label)}" '
        f'style="{style}" vertex="1" parent="1">\n'
        f'          <mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry" />\n'
        "        </mxCell>\n"
    )


def mx_edge(
    cell_id: str,
    source: str,
    target: str,
    *,
    style: str | None = None,
    label: str = "",
) -> str:
    edge_style = style or (
        "edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;"
        "html=1;strokeColor=#111111;strokeWidth=2;endArrow=block;endFill=1;"
    )
    return (
        f'        <mxCell id="{cell_id}" value="{html_label(label)}" '
        f'style="{edge_style}" edge="1" parent="1" source="{source}" target="{target}">\n'
        '          <mxGeometry relative="1" as="geometry" />\n'
        "        </mxCell>\n"
    )


def write_drawio(path: Path) -> None:
    font = "fontFamily=Times New Roman;fontSize=22;fontColor=#111111;"
    process = (
        "rounded=0;whiteSpace=wrap;html=1;strokeColor=#111111;strokeWidth=2;"
        f"fillColor=#FFFFFF;spacing=8;{font}"
    )
    output_style = (
        "ellipse;whiteSpace=wrap;html=1;strokeColor=#111111;strokeWidth=2;"
        f"fillColor=#FFFFFF;spacing=8;{font}"
    )
    note_style = (
        "rounded=1;whiteSpace=wrap;html=1;strokeColor=#111111;strokeWidth=2;"
        "fillColor=#F5F5F5;dashed=1;spacing=8;"
        f"{font}"
    )
    text_style = (
        "text;html=1;align=center;verticalAlign=middle;resizable=0;points=[];"
        "autosize=1;fontFamily=Times New Roman;fontSize=21;fontColor=#111111;"
    )
    dashed_edge = (
        "edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;"
        "html=1;strokeColor=#111111;strokeWidth=1.5;dashed=1;endArrow=block;endFill=1;"
    )

    cells = [
        mx_vertex("raw", "Raw codes\nDc, F", process, 35, 155, 155, 72),
        mx_vertex("intra", "Intra-group\nalign", process, 245, 155, 175, 72),
        mx_vertex("streams", "A/B streams\nUA, UB", process, 475, 155, 150, 72),
        mx_vertex("predict", "Predict\nDhatB", process, 680, 155, 150, 72),
        mx_vertex("target", "Fine target\nFhatB", process, 885, 155, 150, 72),
        mx_vertex("fit", "Fit\ngB, oB, kB", process, 1090, 155, 150, 72),
        mx_vertex("output", "Output\nDcorr", output_style, 1295, 155, 155, 72),
        mx_vertex("predictor", "P{UA}", note_style, 680, 45, 150, 58),
        mx_vertex("safe", "Safe\nsamples", note_style, 1085, 270, 160, 58),
        mx_vertex("formula", "FhatB = DhatB - Dc,B", text_style, 870, 238, 180, 30),
        mx_vertex("bypass", "coarse bypass", text_style, 890, 305, 140, 30),
        mx_edge("e1", "raw", "intra"),
        mx_edge("e2", "intra", "streams"),
        mx_edge("e3", "streams", "predict"),
        mx_edge("e4", "predict", "target"),
        mx_edge("e5", "target", "fit"),
        mx_edge("e6", "fit", "output"),
        mx_edge("e7", "predictor", "predict", style=dashed_edge),
        mx_edge("e8", "safe", "fit", style=dashed_edge),
        mx_edge("e9", "streams", "output", style=dashed_edge),
    ]

    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<mxfile host="drawio" version="26.0.0">\n'
        '  <diagram name="Fig. 3">\n'
        '    <mxGraphModel dx="1500" dy="390" grid="1" gridSize="10" guides="1" '
        'tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" '
        'pageWidth="1500" pageHeight="390" math="0" shadow="0">\n'
        "      <root>\n"
        '        <mxCell id="0" />\n'
        '        <mxCell id="1" parent="0" />\n'
        + "".join(cells)
        + "      </root>\n"
        "    </mxGraphModel>\n"
        "  </diagram>\n"
        "</mxfile>\n"
    )
    path.write_text(xml, encoding="utf-8")


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    base = out_dir / "fig3_calibration_flow"
    write_drawio(base.with_suffix(".drawio"))
    fig = build_figure()
    save_figure(fig, base)
    plt.close(fig)


if __name__ == "__main__":
    main()
