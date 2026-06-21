from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import patches


# Editable text for SVG/PDF post-processing.
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42


BLACK = "#111111"
GRAY = "#666666"
LIGHT = "#F5F5F5"


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 7.2,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "savefig.facecolor": "white",
        }
    )


def box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    *,
    fontsize: float = 7.0,
    weight: str = "normal",
    lw: float = 1.05,
    facecolor: str = "white",
    linestyle: str = "-",
) -> None:
    rect = patches.Rectangle(
        (x, y),
        w,
        h,
        linewidth=lw,
        edgecolor=BLACK,
        facecolor=facecolor,
        linestyle=linestyle,
        zorder=2,
    )
    ax.add_patch(rect)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        color=BLACK,
        linespacing=1.10,
        zorder=3,
    )


def text(
    ax,
    x: float,
    y: float,
    s: str,
    *,
    fontsize: float = 6.6,
    weight: str = "normal",
    color: str = BLACK,
    ha: str = "center",
    va: str = "center",
) -> None:
    ax.text(
        x,
        y,
        s,
        ha=ha,
        va=va,
        fontsize=fontsize,
        fontweight=weight,
        color=color,
        linespacing=1.10,
        zorder=4,
    )


def arrow(
    ax,
    p0: tuple[float, float],
    p1: tuple[float, float],
    *,
    lw: float = 1.05,
    color: str = BLACK,
    mutation_scale: float = 8.5,
) -> None:
    ax.add_patch(
        patches.FancyArrowPatch(
            p0,
            p1,
            arrowstyle="-|>",
            mutation_scale=mutation_scale,
            linewidth=lw,
            color=color,
            shrinkA=0,
            shrinkB=0,
            zorder=1,
        )
    )


def line(ax, p0: tuple[float, float], p1: tuple[float, float], *, lw: float = 1.05) -> None:
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=BLACK, lw=lw, zorder=1)


def elbow_arrow(
    ax,
    pts: list[tuple[float, float]],
    *,
    lw: float = 1.05,
    color: str = BLACK,
    mutation_scale: float = 8.5,
) -> None:
    for p0, p1 in zip(pts[:-2], pts[1:-1]):
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=color, lw=lw, zorder=1)
    arrow(ax, pts[-2], pts[-1], lw=lw, color=color, mutation_scale=mutation_scale)


def panel_label(ax, label: str, title: str) -> None:
    text(ax, 0.015, 0.955, label, fontsize=8.5, weight="bold", ha="left")
    text(ax, 0.055, 0.955, title, fontsize=8.0, weight="bold", ha="left")


def prepare_axis(ax) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


def draw_general_architecture(ax) -> None:
    prepare_axis(ax)
    panel_label(ax, "a", "General partially interleaved pipelined-SAR ADC architecture")

    # Front-end and coarse code path.
    box(ax, 0.035, 0.665, 0.100, 0.115, "Analog\ninput")
    box(ax, 0.165, 0.665, 0.085, 0.115, "S/H", weight="bold")
    box(ax, 0.295, 0.625, 0.150, 0.195, "Coarse\nstage", weight="bold")
    box(ax, 0.830, 0.665, 0.130, 0.115, "Digital\ncombiner", weight="bold")

    arrow(ax, (0.135, 0.722), (0.165, 0.722))
    arrow(ax, (0.250, 0.722), (0.295, 0.722))
    arrow(ax, (0.445, 0.735), (0.830, 0.735))
    text(ax, 0.635, 0.770, "Coarse code", fontsize=6.8, color=GRAY)

    # Generic residue / fine-code branches.
    line(ax, (0.370, 0.625), (0.370, 0.275))
    elbow_arrow(ax, [(0.370, 0.515), (0.485, 0.515)])
    elbow_arrow(ax, [(0.370, 0.345), (0.485, 0.345)])

    box(ax, 0.485, 0.465, 0.110, 0.100, "MDAC\nGroup 1", fontsize=6.6, weight="bold")
    box(ax, 0.485, 0.295, 0.110, 0.100, "MDAC\nGroup M", fontsize=6.6, weight="bold")
    text(ax, 0.540, 0.425, "...", fontsize=8.5)

    # Each SAR is an independent small box.
    sar_w, sar_h = 0.088, 0.062
    sar_x = 0.650
    for y, name in [(0.518, "SAR 1.1"), (0.438, "SAR 1.2"), (0.348, "SAR M.1"), (0.268, "SAR M.2")]:
        box(ax, sar_x, y, sar_w, sar_h, name, fontsize=6.1)

    # Right-angle fan-out from each MDAC group to separate SAR boxes.
    split_x = 0.625
    line(ax, (0.595, 0.515), (split_x, 0.515))
    line(ax, (split_x, 0.469), (split_x, 0.549))
    arrow(ax, (split_x, 0.549), (sar_x, 0.549))
    arrow(ax, (split_x, 0.469), (sar_x, 0.469))
    line(ax, (0.595, 0.345), (split_x, 0.345))
    line(ax, (split_x, 0.299), (split_x, 0.379))
    arrow(ax, (split_x, 0.379), (sar_x, 0.379))
    arrow(ax, (split_x, 0.299), (sar_x, 0.299))

    # Fine code bus into digital combiner.
    bus_x = 0.775
    line(ax, (sar_x + sar_w, 0.549), (bus_x, 0.549))
    line(ax, (sar_x + sar_w, 0.469), (bus_x, 0.469))
    line(ax, (sar_x + sar_w, 0.379), (bus_x, 0.379))
    line(ax, (sar_x + sar_w, 0.299), (bus_x, 0.299))
    line(ax, (bus_x, 0.299), (bus_x, 0.515))
    elbow_arrow(ax, [(bus_x, 0.515), (0.895, 0.515), (0.895, 0.665)])
    text(ax, 0.835, 0.555, "Fine code", fontsize=6.8, color=GRAY)


def draw_representative_architecture(ax) -> None:
    prepare_axis(ax)
    panel_label(ax, "b", "Representative 2-MDAC / 4-SAR implementation used for verification")

    box(ax, 0.035, 0.650, 0.100, 0.115, "Analog\ninput")
    box(ax, 0.165, 0.650, 0.085, 0.115, "S/H", weight="bold")
    box(ax, 0.295, 0.605, 0.150, 0.205, "2.5-bit Flash\n+ coarse DAC", fontsize=6.8, weight="bold")
    box(ax, 0.830, 0.650, 0.130, 0.115, "Digital\ncombiner", weight="bold")

    arrow(ax, (0.135, 0.708), (0.165, 0.708))
    arrow(ax, (0.250, 0.708), (0.295, 0.708))
    arrow(ax, (0.445, 0.725), (0.830, 0.725))
    text(ax, 0.635, 0.760, "Coarse code", fontsize=6.8, color=GRAY)

    # Two MDAC branches.
    line(ax, (0.370, 0.605), (0.370, 0.245))
    elbow_arrow(ax, [(0.370, 0.505), (0.500, 0.505)])
    elbow_arrow(ax, [(0.370, 0.330), (0.500, 0.330)])
    box(ax, 0.500, 0.455, 0.105, 0.100, "MDAC-A", weight="bold")
    box(ax, 0.500, 0.280, 0.105, 0.100, "MDAC-B", weight="bold")

    # Four independent SAR boxes.
    sar_w, sar_h = 0.090, 0.064
    sar_x = 0.665
    sar_positions = [
        (0.540, "SAR ch0"),
        (0.455, "SAR ch2"),
        (0.330, "SAR ch1"),
        (0.245, "SAR ch3"),
    ]
    for y, name in sar_positions:
        box(ax, sar_x, y, sar_w, sar_h, name, fontsize=6.2)

    # MDAC-A to ch0/ch2; MDAC-B to ch1/ch3, with right-angle fan-out.
    split_x = 0.635
    line(ax, (0.605, 0.505), (split_x, 0.505))
    line(ax, (split_x, 0.487), (split_x, 0.572))
    arrow(ax, (split_x, 0.572), (sar_x, 0.572))
    arrow(ax, (split_x, 0.487), (sar_x, 0.487))
    line(ax, (0.605, 0.330), (split_x, 0.330))
    line(ax, (split_x, 0.277), (split_x, 0.362))
    arrow(ax, (split_x, 0.362), (sar_x, 0.362))
    arrow(ax, (split_x, 0.277), (sar_x, 0.277))

    # Fine code bus and combiner input.
    bus_x = 0.775
    for y in [0.572, 0.487, 0.362, 0.277]:
        line(ax, (sar_x + sar_w, y), (bus_x, y))
    line(ax, (bus_x, 0.277), (bus_x, 0.525))
    elbow_arrow(ax, [(bus_x, 0.525), (0.895, 0.525), (0.895, 0.650)])
    text(ax, 0.835, 0.565, "Fine code", fontsize=6.8, color=GRAY)

    # Output arrow.
    arrow(ax, (0.960, 0.708), (0.990, 0.708))


def build_figure() -> plt.Figure:
    setup_style()
    fig = plt.figure(figsize=(7.2, 4.6), constrained_layout=False)
    gs = fig.add_gridspec(2, 1, hspace=0.12)
    draw_general_architecture(fig.add_subplot(gs[0, 0]))
    draw_representative_architecture(fig.add_subplot(gs[1, 0]))
    fig.subplots_adjust(left=0.015, right=0.985, top=0.970, bottom=0.045)
    return fig


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    fig = build_figure()
    base = out_dir / "fig1_architecture"
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(
        base.with_suffix(".tiff"),
        dpi=600,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
