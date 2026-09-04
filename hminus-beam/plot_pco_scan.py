"""
Plot a scan of ELENA settings against the PCO Edge images.

Two views, both driven by the same per-run parquet cache (see hminus_data.py):

  grid    rows and columns are the values of two knobs, each cell is that run's PCO image
  matrix  every scanned knob on both axes, each cell a scatter coloured by an image observable

Axes default to the **requested setpoints** (ELENA_QD1_V, ELENA_H1_CORRECTOR_V, ...) rather than the
ELENA_Parameters readbacks, because the setpoint is the exact number the scan asked for and needs no
binning. Readbacks still work as axes and are binned to BIN_WIDTH; hminus_data.REQUESTED_TO_DEVICE
pairs each setpoint with the channel it lands on.

Only the driven channels have a setpoint: *R on the horizontal dipoles, *T on the vertical dipoles
and *P on the quadrupoles. The matching *L / *B / *N channels read back as their negatives and carry
no extra information -- check_mirror_channels() verifies that per run.

    python plot_pco_scan.py --runs 523357-523383
    python plot_pco_scan.py --runs 523357-523383 --grid ELENA_QF1_V ELENA_QD1_V
"""

import argparse
import os

import numpy as np
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

import hminus_data as hd

# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
RUNS = [492712, 492714, 492715, 492716]

# corrector readbacks jitter by ~0.01 between runs while real scan steps are tens to hundreds, so
# group settings into bins of this width instead of comparing raw floats
BIN_WIDTH = hd.BIN_WIDTH

# grid view
ROW_CORRECTOR = None     # None -> pick the first two correctors that actually varied
COL_CORRECTOR = None
CELL_PX = 200
GRID_DISPLAY_SIZE = 256  # images are small in a grid, so downsample harder than in the panel plot
AGGREGATE = "mean"      # "first" or "mean", used when several runs land in the same cell
DOWNSAMPLE_HOW = "max"   # peak-preserving; see hminus_data.downsample
COLORSCALE = "Magma"
# background is 0 after ALPACA's subtraction and the beam covers few pixels, so anchor at 0 and
# scale to the peak rather than to a percentile (which would saturate on noise)
ZMIN = 0.0
CONTRAST = 1.0           # zmax = peak / CONTRAST; raise it to bring out faint structure

# matrix view
OBSERVABLE = "PCO_signal_sum"
DIAGONAL = "response"    # "response" (corrector vs observable) or "histogram"
DROP_CONSTANT = True     # hide correctors that never changed across the run list
MATRIX_CELL_PX = 170


def _binned(data: pl.DataFrame, column: str) -> pl.Series:
    """Grid coordinate for a column: exact for a requested setpoint, binned for a readback."""
    values = data.get_column(column)
    if column in hd.SETPOINTS:
        return values

    return hd.bin_values(values, BIN_WIDTH)


def _varying(data: pl.DataFrame, columns: list[str]) -> list[str]:
    """Which of these columns the run set actually scanned."""
    setpoints = [name for name in columns if name in hd.SETPOINTS]
    readbacks = [name for name in columns if name not in hd.SETPOINTS]

    return (hd.varying_setpoints(data, setpoints)
            + hd.varying_correctors(data, readbacks, bin_width=BIN_WIDTH))


def _colour_range(images: list[np.ndarray]) -> tuple[float, float]:
    peak = max(float(image.max()) for image in images)
    return ZMIN, peak / CONTRAST if CONTRAST else peak


def _label(column: str) -> str:
    """Name a column by both of its identities where it has two."""
    if column in hd.SETPOINTS:
        return f"{column} (set, reads back on {hd.REQUESTED_TO_DEVICE[column]})"
    short = hd.DEVICE_CORRECTOR.get(column)
    return f"{column} ({short})" if short else column


def _axis_names(index: int) -> tuple[str, str]:
    suffix = "" if index == 1 else str(index)
    return f"x{suffix}", f"y{suffix}"


# ---------------------------------------------------------------------------
# (a) image grid: rows and columns are two correctors
# ---------------------------------------------------------------------------
def plot_scan_grid(data: pl.DataFrame,
                   row_corrector: str,
                   col_corrector: str,
                   display_size: int = GRID_DISPLAY_SIZE,
                   aggregate: str = AGGREGATE,
                   output_name: str | None = None,
                   show: bool = True) -> go.Figure:
    """Lay the PCO images out on a grid spanned by two ELENA correctors."""
    for corrector in (row_corrector, col_corrector):
        if corrector not in data.columns:
            raise KeyError(f"{corrector} is not a column; use one of the requested setpoints "
                           f"{hd.SETPOINTS} or the readbacks {hd.SET_CORRECTORS}")

    data = data.filter(pl.col("has_image")
                       & pl.col(row_corrector).is_not_null()
                       & pl.col(col_corrector).is_not_null())
    if data.is_empty():
        raise RuntimeError(f"no run has an image plus both {row_corrector} and {col_corrector}")

    data = data.with_columns(_binned(data, row_corrector).alias("_row_value"),
                             _binned(data, col_corrector).alias("_col_value"))

    # higher row values at the top, matching the convention in ELENA_beam_steering
    row_values = sorted(data.get_column("_row_value").unique().to_list(), reverse=True)
    col_values = sorted(data.get_column("_col_value").unique().to_list())

    cells: dict[tuple[float, float], list[dict]] = {}
    for row in data.iter_rows(named=True):
        cells.setdefault((row["_row_value"], row["_col_value"]), []).append(row)

    fig = make_subplots(
        rows=len(row_values), cols=len(col_values),
        row_titles=[f"{value:g}" for value in row_values],
        column_titles=[f"{value:g}" for value in col_values],
        horizontal_spacing=0.01, vertical_spacing=0.015,
        # Not shared: each cell is an independent image in its own pixel coordinates, and sharing
        # would make plotly reuse axis objects so empty cells have no axis to anchor a label to.
        shared_xaxes=False, shared_yaxes=False)

    prepared, index = {}, 0
    for grid_row, row_value in enumerate(row_values, start=1):
        for grid_col, col_value in enumerate(col_values, start=1):
            index += 1
            matches = cells.get((row_value, col_value))
            if not matches:
                continue

            images = [hd.downsample(hd.get_image(match), display_size, DOWNSAMPLE_HOW)
                      for match in matches]
            image = np.mean(images, axis=0) if aggregate == "mean" and len(images) > 1 else images[0]
            prepared[index] = (grid_row, grid_col, matches, image)

    if not prepared:
        raise RuntimeError("no populated cells in the scan grid")

    zmin, zmax = _colour_range([image for *_, image in prepared.values()])

    for index, (grid_row, grid_col, matches, image) in prepared.items():
        runs = [match["Run Number"] for match in matches]
        shown = "mean of " if aggregate == "mean" and len(runs) > 1 else ""
        description = hd.describe_elena(matches[0])

        fig.add_trace(
            go.Heatmap(z=image, coloraxis="coloraxis",
                       hovertemplate=(f"{_label(row_corrector)} = {matches[0]['_row_value']:g}"
                                      f"<br>{_label(col_corrector)} = {matches[0]['_col_value']:g}"
                                      "<br>signal=%{z:.4g}"
                                      f"<br><br>{description}"
                                      f"<extra>{shown}runs {runs}</extra>")),
            row=grid_row, col=grid_col)

        x_axis, _ = _axis_names(index)
        fig.update_yaxes(autorange="reversed", scaleanchor=x_axis, constrain="domain",
                         row=grid_row, col=grid_col)

    for grid_row, row_value in enumerate(row_values, start=1):
        for grid_col, col_value in enumerate(col_values, start=1):
            if (row_value, col_value) in cells:
                continue
            # A subplot with no trace is not rendered at all, so its frame and any annotation
            # anchored to it silently disappear. An invisible point gives the cell something to
            # draw. make_subplots numbers axes row-major, so the axis pair is predictable.
            fig.add_trace(go.Scatter(x=[0.5], y=[0.5], mode="markers", showlegend=False,
                                     marker=dict(opacity=0), hoverinfo="skip"),
                          row=grid_row, col=grid_col)
            fig.update_xaxes(range=[0, 1], row=grid_row, col=grid_col)
            fig.update_yaxes(range=[0, 1], row=grid_row, col=grid_col)

            x_axis, y_axis = _axis_names((grid_row - 1) * len(col_values) + grid_col)
            fig.add_annotation(text="no run", showarrow=False,
                               font=dict(size=9, color="#bbbbbb"),
                               xref=f"{x_axis} domain", yref=f"{y_axis} domain", x=0.5, y=0.5)

    # one light frame per cell, so empty positions in the scan still read as cells
    fig.update_xaxes(showticklabels=False, ticks="", showgrid=False, zeroline=False,
                     showline=True, linecolor="#dddddd", mirror=True)
    fig.update_yaxes(showticklabels=False, ticks="", showgrid=False, zeroline=False,
                     showline=True, linecolor="#dddddd", mirror=True)

    empty = len(row_values) * len(col_values) - len(prepared)
    # exact requested setpoints need no binning; noisy readbacks do
    binned_axes = [name for name in (row_corrector, col_corrector) if name not in hd.SETPOINTS]
    binning_note = (f", {' and '.join(binned_axes)} binned to steps of {BIN_WIDTH:g}"
                    if binned_axes else ", exact requested setpoints")
    fig.update_layout(
        title=(f"PCO Edge scan: rows {_label(row_corrector)}, columns {_label(col_corrector)}"
               f"<br><sub>{data.height} runs, {len(prepared)} populated cells, {empty} empty"
               f"{binning_note}</sub>"),
        height=CELL_PX * len(row_values) + 190,
        width=CELL_PX * len(col_values) + 220,
        template="plotly_white",
        coloraxis=dict(colorscale=COLORSCALE, cmin=zmin, cmax=zmax,
                       colorbar=dict(title="signal", thickness=14)),
        margin=dict(t=120, b=70, l=90, r=90))

    os.makedirs(hd.PLOT_DIR, exist_ok=True)
    if output_name is None:
        output_name = f"pco_scan_{row_corrector}_vs_{col_corrector}.html"
    path = os.path.join(hd.PLOT_DIR, output_name)
    fig.write_html(path)
    print(f"wrote {path} ({os.path.getsize(path) / 1e6:.1f} MB)")

    if show:
        fig.show()

    return fig


# ---------------------------------------------------------------------------
# (b) corrector matrix: correctors on both axes
# ---------------------------------------------------------------------------
def plot_corrector_matrix(data: pl.DataFrame,
                          correctors: list[str] | None = None,
                          observable: str = OBSERVABLE,
                          diagonal: str = DIAGONAL,
                          drop_constant: bool = DROP_CONSTANT,
                          output_name: str | None = None,
                          show: bool = True) -> go.Figure:
    """Pair plot of the driven correctors, coloured by an image observable."""
    if correctors is None:
        correctors = hd.SETPOINTS
    correctors = [c for c in correctors if c in data.columns]
    if drop_constant:
        varying = _varying(data, correctors)
        if varying:
            correctors = varying
        else:
            print("no corrector varied across these runs; keeping all of them")
    if not correctors:
        raise RuntimeError("no corrector columns to plot")
    if observable not in data.columns:
        raise KeyError(f"{observable} is not a column of the data")

    data = data.filter(pl.col(observable).is_not_null())
    runs = data.get_column("Run Number").to_list()
    values = {corrector: data.get_column(corrector).to_list() for corrector in correctors}
    observable_values = data.get_column(observable).to_list()

    size = len(correctors)
    fig = make_subplots(rows=size, cols=size,
                        horizontal_spacing=0.012, vertical_spacing=0.012,
                        shared_xaxes=True, shared_yaxes=False)

    for grid_row, y_corrector in enumerate(correctors, start=1):
        for grid_col, x_corrector in enumerate(correctors, start=1):
            on_diagonal = grid_row == grid_col

            if on_diagonal and diagonal == "histogram":
                fig.add_trace(go.Histogram(x=values[x_corrector], marker_color="#666666",
                                           showlegend=False,
                                           hovertemplate=f"{x_corrector}=%{{x}}<br>runs=%{{y}}"
                                                         "<extra></extra>"),
                              row=grid_row, col=grid_col)
            else:
                y_values = observable_values if on_diagonal else values[y_corrector]
                y_name = observable if on_diagonal else y_corrector
                fig.add_trace(
                    go.Scatter(x=values[x_corrector], y=y_values, mode="markers",
                               customdata=list(zip(runs, observable_values)), showlegend=False,
                               marker=dict(size=7, coloraxis="coloraxis",
                                           color=observable_values,
                                           line=dict(width=0.5, color="#ffffff")),
                               hovertemplate=(f"{x_corrector}=%{{x:.4g}}<br>{y_name}=%{{y:.4g}}"
                                              f"<br>{observable}=%{{customdata[1]:.4g}}"
                                              "<extra>run %{customdata[0]}</extra>")),
                    row=grid_row, col=grid_col)

            if grid_row == size:
                fig.update_xaxes(title_text=x_corrector, title_font=dict(size=9),
                                 row=grid_row, col=grid_col)
            if grid_col == 1:
                fig.update_yaxes(title_text=y_corrector, title_font=dict(size=9),
                                 row=grid_row, col=grid_col)
            fig.update_xaxes(showticklabels=False, ticks="", row=grid_row, col=grid_col)
            fig.update_yaxes(showticklabels=False, ticks="", row=grid_row, col=grid_col)

    diagonal_note = ("histogram of the sampled values" if diagonal == "histogram"
                     else f"corrector vs {observable}")
    fig.update_layout(
        title=(f"ELENA corrector scan matrix, {data.height} runs"
               f"<br><sub>marker colour = {observable}; diagonal = {diagonal_note}</sub>"),
        height=MATRIX_CELL_PX * size + 190,
        width=MATRIX_CELL_PX * size + 210,
        template="plotly_white",
        coloraxis=dict(colorscale="Viridis", colorbar=dict(title=observable, thickness=14)),
        margin=dict(t=110, b=80, l=110, r=90))

    os.makedirs(hd.PLOT_DIR, exist_ok=True)
    if output_name is None:
        output_name = "pco_corrector_matrix.html"
    path = os.path.join(hd.PLOT_DIR, output_name)
    fig.write_html(path)
    print(f"wrote {path} ({os.path.getsize(path) / 1e6:.1f} MB)")

    if show:
        fig.show()

    return fig


def _parse_runs(tokens: list[str]) -> list[int]:
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
    parser.add_argument("--grid", nargs=2, metavar=("ROW", "COL"),
                        default=[ROW_CORRECTOR, COL_CORRECTOR],
                        help="correctors for the grid rows and columns (default: first two varying)")
    parser.add_argument("--observable", default=OBSERVABLE, help="colour axis of the matrix")
    parser.add_argument("--diagonal", choices=["response", "histogram"], default=DIAGONAL)
    parser.add_argument("--skip-grid", action="store_true")
    parser.add_argument("--skip-matrix", action="store_true")
    parser.add_argument("--force", action="store_true", help="re-download even if cached")
    parser.add_argument("--no-show", action="store_true", help="only write the html files")
    args = parser.parse_args()

    runs = _parse_runs(args.runs) if args.runs else RUNS
    data = hd.load_runs(runs, force=args.force)

    offenders = hd.check_mirror_channels(data)
    if offenders.is_empty():
        print("mirror check: every *L / *B / *N channel is the negative of its driven partner")
    else:
        print(f"mirror check: {offenders.height} channel readbacks are NOT mirrored -- the readback "
              f"was probably taken while the corrector was still ramping:")
        print(offenders)

    varying = _varying(data, hd.SETPOINTS)
    print(f"setpoints that varied across these runs: {varying or 'none'}")
    varying_readbacks = _varying(data, hd.SET_CORRECTORS)
    print(f"readbacks that varied across these runs: {varying_readbacks or 'none'}")

    if not args.skip_matrix:
        plot_corrector_matrix(data, observable=args.observable, diagonal=args.diagonal,
                              show=not args.no_show)

    if not args.skip_grid:
        row_corrector, col_corrector = args.grid
        if row_corrector is None or col_corrector is None:
            if len(varying) < 2:
                print(f"skipping the grid: it needs two knobs that varied, found {varying}")
                raise SystemExit(0)
            row_corrector, col_corrector = varying[0], varying[1]
            print(f"grid correctors not given, using {row_corrector} x {col_corrector}")

        plot_scan_grid(data, row_corrector, col_corrector, show=not args.no_show)
