"""
Plot every PCO Edge picture from a list of runs, one panel per run, annotated with the ELENA
settings of that run.

Data comes from the local per-run parquet cache (see hminus_data.py); runs that are not cached yet
are downloaded through ALPACA on first use.

    python plot_pco_images.py
    python plot_pco_images.py --runs 492712-492719 492730
"""

import argparse
import os

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import hminus_data as hd

# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
RUNS = [int(run) for run in np.linspace(523357,523410,54)]
print(RUNS)

N_COLS = 3
PANEL_PX = 380           # rendered size of one panel, in pixels
DISPLAY_SIZE = 512       # block-reduce the 1024x1024 image to this before plotting
DOWNSAMPLE_HOW = "max"   # peak-preserving; the beam spot covers too few pixels to survive a mean
COLORSCALE = "Magma"
COMMON_SCALE = True      # one colour range shared by every panel

# The PCO Edge background sits at 0 after ALPACA's row-wise subtraction and the beam occupies a tiny
# fraction of the frame (median ~0, 99.99th percentile ~2, max in the hundreds), so a percentile
# range would just saturate on noise. Anchor at 0 and scale to the peak, as ALPACA's own image
# viewer does (applications/alpaca_image_viewer/app.py).
ZMIN = 0.0
CONTRAST = 1.0           # zmax = peak / CONTRAST; raise it to bring out faint structure

# ELENA keys written under each panel; the full set is always in the hover text
ANNOTATION_KEYS = ["H_offset_mm", "V_offset_mm", "H_angle_mrad", "V_angle_mrad", "catch_delay"]


def _axis_names(index: int) -> tuple[str, str]:
    """Plotly names the first pair of axes 'x'/'y' and the rest 'x2'/'y2', ..."""
    suffix = "" if index == 1 else str(index)
    return f"x{suffix}", f"y{suffix}"


def _colour_range(images: list[np.ndarray], contrast: float) -> tuple[float, float]:
    peak = max(float(image.max()) for image in images)
    return ZMIN, peak / contrast if contrast else peak


def plot_run_images(data,
                    n_cols: int = N_COLS,
                    display_size: int = DISPLAY_SIZE,
                    common_scale: bool = COMMON_SCALE,
                    contrast: float = CONTRAST,
                    output_name: str | None = None,
                    show: bool = True) -> go.Figure:
    """One panel per run, each showing that run's PCO Edge image plus its ELENA settings."""
    rows = [row for row in data.iter_rows(named=True) if row.get("has_image")]
    skipped = [row["Run Number"] for row in data.iter_rows(named=True) if not row.get("has_image")]
    if skipped:
        print(f"no PCO Edge image for runs {skipped}, they get no panel")
    if not rows:
        raise RuntimeError("none of the requested runs has a PCO Edge image")

    images = [hd.downsample(hd.get_image(row), display_size, DOWNSAMPLE_HOW)
              for row in rows]

    n_cols = max(1, min(n_cols, len(rows)))
    n_rows = int(np.ceil(len(rows) / n_cols))

    fig = make_subplots(rows=n_rows, cols=n_cols,
                        subplot_titles=[f"run {row['Run Number']}" for row in rows],
                        horizontal_spacing=0.04,
                        vertical_spacing=max(0.06, 0.22 / n_rows))

    zmin, zmax = _colour_range(images, contrast) if common_scale else (None, None)

    for index, (row, image) in enumerate(zip(rows, images), start=1):
        grid_row, grid_col = divmod(index - 1, n_cols)
        description = hd.describe_elena(row)

        heatmap = go.Heatmap(
            z=image,
            name=str(row["Run Number"]),
            hovertemplate=("x=%{x} px<br>y=%{y} px<br>signal=%{z:.4g}"
                           f"<br><br>{description}<extra>run {row['Run Number']}</extra>"),
        )
        if common_scale:
            heatmap.update(coloraxis="coloraxis")
        else:
            heatmap.update(colorscale=COLORSCALE, showscale=False,
                           zmin=ZMIN, zmax=float(image.max()) / (contrast or 1))

        fig.add_trace(heatmap, row=grid_row + 1, col=grid_col + 1)

        x_axis, y_axis = _axis_names(index)
        fig.update_xaxes(showticklabels=False, ticks="", showgrid=False, zeroline=False,
                         row=grid_row + 1, col=grid_col + 1)
        # reversed y so row 0 of the array is drawn at the top, as in the raw frame
        fig.update_yaxes(showticklabels=False, ticks="", showgrid=False, zeroline=False,
                         autorange="reversed", scaleanchor=x_axis, constrain="domain",
                         row=grid_row + 1, col=grid_col + 1)

        fig.add_annotation(
            text=hd.describe_elena(row, keys=ANNOTATION_KEYS, sep="<br>"),
            xref=f"{x_axis} domain", yref=f"{y_axis} domain",
            x=0, y=-0.02, xanchor="left", yanchor="top",
            showarrow=False, align="left",
            font=dict(size=9, family="monospace"),
        )

    first, last = rows[0]["Run Number"], rows[-1]["Run Number"]
    fig.update_layout(
        title=f"PCO Edge images, runs {first}-{last}",
        height=PANEL_PX * n_rows + 140,
        width=PANEL_PX * n_cols + 160,
        template="plotly_white",
        margin=dict(t=90, b=60, l=60, r=60),
    )
    if common_scale:
        fig.update_layout(coloraxis=dict(colorscale=COLORSCALE, cmin=zmin, cmax=zmax,
                                         colorbar=dict(title="signal", thickness=14)))
    for annotation in fig.layout.annotations:
        if annotation.text.startswith("run "):
            annotation.font.size = 12

    os.makedirs(hd.PLOT_DIR, exist_ok=True)
    if output_name is None:
        output_name = f"pco_images_{first}-{last}.html"
    path = os.path.join(hd.PLOT_DIR, output_name)
    fig.write_html(path)
    print(f"wrote {path} ({os.path.getsize(path) / 1e6:.1f} MB)")

    if show:
        fig.show()

    return fig


def _parse_runs(tokens: list[str]) -> list[int]:
    """Accept both single runs and inclusive 'first-last' ranges."""
    runs = []
    for token in tokens:
        if "-" in token.strip("-"):
            first, last = token.split("-", 1)
            runs.extend(range(int(first), int(last) + 1))
        else:
            runs.append(int(token))

    return sorted(set(runs))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", default=None,
                        help="run numbers and/or first-last ranges (default: the RUNS constant)")
    parser.add_argument("--cols", type=int, default=N_COLS, help="panels per row")
    parser.add_argument("--display-size", type=int, default=DISPLAY_SIZE,
                        help="downsample images to this size before plotting")
    parser.add_argument("--per-panel-scale", action="store_true",
                        help="give each panel its own colour range instead of a shared one")
    parser.add_argument("--contrast", type=float, default=CONTRAST,
                        help="zmax = peak / contrast; raise it to bring out faint structure")
    parser.add_argument("--force", action="store_true", help="re-download even if cached")
    parser.add_argument("--no-show", action="store_true", help="only write the html file")
    args = parser.parse_args()

    runs = _parse_runs(args.runs) if args.runs else RUNS

    data = hd.load_runs(runs, force=args.force)
    plot_run_images(data,
                    n_cols=args.cols,
                    display_size=args.display_size,
                    common_scale=not args.per_panel_scale,
                    contrast=args.contrast,
                    show=not args.no_show)
