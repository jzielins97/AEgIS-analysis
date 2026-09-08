"""
Plot a scan of ELENA settings against the PCO Edge images.

Two views, both driven by the same per-run parquet cache (see hminus_data.py):

  grid    rows and columns are the values of two knobs, each cell is that run's PCO image
  matrix  every scanned knob on both axes, each cell a scatter coloured by an image observable

A knob here is either a corrector setpoint (ELENA_QD1_V, ELENA_H1_CORRECTOR_V, ...) or one of the
four ELENA steering knobs (H_offset_requested, V_angle_requested, ...). ELENA can be steered either
way, and a run set that was steered by the knobs leaves every corrector setpoint sitting still, so
restricting the axes to correctors makes such a scan look like no scan at all.

Axes default to the **requested** values rather than the ELENA_Parameters readbacks, because the
request is the exact number the scan asked for and needs no binning. Readbacks still work as axes
and are binned per unit (hminus_data.bin_width_for); hminus_data.SETPOINT_TO_READBACK pairs each
request with the channel it lands on.

Only the driven channels have a setpoint: *R on the horizontal dipoles, *T on the vertical dipoles
and *P on the quadrupoles. The matching *L / *B / *N channels read back as their negatives and carry
no extra information -- check_mirror_channels() verifies that per run.

    python plot_pco_scan.py --runs 523357-523383
    python plot_pco_scan.py --runs 523357-523383 --grid ELENA_QF1_V ELENA_QD1_V
    python plot_pco_scan.py --runs 523415-523441
    python plot_pco_scan.py --runs 523415-523441 --grid V_offset_requested H_offset_requested
    python plot_pco_scan.py --runs 523415-523441 --axes H_offset_requested DHZE08R DVTE14T
"""

import argparse
import os
import textwrap

import numpy as np
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

import hminus_data as hd

# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
RUNS = [492712, 492714, 492715, 492716]

# grid view. Readback axes are binned, at a width that follows the column's unit -- volts for a
# corrector, mm or mrad for a steering knob. See hminus_data.bin_width_for.
ROW_KNOB = None          # None -> pick the first two knobs that actually varied
COL_KNOB = None
CELL_PX = 200
GRID_DISPLAY_SIZE = 256  # images are small in a grid, so downsample harder than in the panel plot
AGGREGATES = ("mean", "first")   # how repeats of one setting are combined into a cell
AGGREGATE = "mean"
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
    """Grid coordinate for a column: exact for a requested value, binned for a readback.

    The bin width follows the column's unit. The corrector's 1 V applied to H_offset_mm would fold a
    whole sub-millimetre scan into a single cell.
    """
    values = data.get_column(column)
    if column in hd.ALL_SETPOINTS:
        return values

    return hd.bin_values(values, hd.bin_width_for(column))


def _varying(data: pl.DataFrame, columns: list[str]) -> list[str]:
    """Which of these columns the run set actually scanned, requests exactly and readbacks binned."""
    setpoints = [name for name in columns if name in hd.ALL_SETPOINTS]
    readbacks = [name for name in columns if name not in hd.ALL_SETPOINTS]

    return hd.varying_setpoints(data, setpoints) + hd.varying_readbacks(data, readbacks)


def _colour_range(images: list[np.ndarray]) -> tuple[float, float]:
    peak = max(float(image.max()) for image in images)
    return ZMIN, peak / CONTRAST if CONTRAST else peak


def _label(column: str) -> str:
    """Name a column by both of its identities where it has two, and always by its unit."""
    unit = hd.KNOB_UNIT.get(column, "")
    # ELENA_QD1_V and H_offset_mm carry their unit in the name already; DHZE08R and
    # H_offset_requested do not, and "H_offset_requested = 2" is ambiguous without one
    suffix = f" [{unit}]" if unit and not column.endswith(f"_{unit}") else ""

    readback = hd.SETPOINT_TO_READBACK.get(column)
    if readback is not None:
        return f"{column}{suffix} (set, reads back on {readback})"

    short = hd.DEVICE_CORRECTOR.get(column)
    return f"{column} ({short}){suffix}" if short else f"{column}{suffix}"


# Plotly clips a title that is wider than the figure instead of wrapping it, and a grid is only
# CELL_PX per column wide. Naming two knobs, their units and the channels they read back on runs to
# roughly 1200 px against an 820 px figure, so the wrapping has to happen here.
TITLE_FONT_PX = 17
SUBTITLE_FONT_PX = 13
TITLE_PAD_PX = 16
# the column knob's name sits in a band under the title block, one AXIS_NAME_LINE_PX per line
AXIS_NAME_FONT_PX = 12
AXIS_NAME_LINE_PX = 16
# rough mean glyph width for the default sans font. It only picks a wrap column, so being a few
# percent out moves a word between lines rather than breaking anything.
CHAR_ASPECT = 0.5


def _wrap(text: str, pixels: int, font_px: int) -> list[str]:
    """Break ``text`` into lines that fit ``pixels`` at ``font_px``."""
    columns = max(20, int(pixels / (font_px * CHAR_ASPECT)))

    return textwrap.wrap(text, width=columns) or [text]


def _title(main: str, subtitle: str, width: int, side_margins: int) -> tuple[str, int]:
    """Title markup wrapped to the figure width, and the top margin the result needs.

    Returning the margin alongside the text is what stops a wrapped title from growing into the
    first row of cells; the caller has no other way to know how many lines it ended up with.
    """
    usable = width - side_margins
    lines = _wrap(main, usable, TITLE_FONT_PX)
    sub_lines = _wrap(subtitle, usable, SUBTITLE_FONT_PX)

    text = ("<br>".join(lines)
            + "<br><sub>" + "<br>".join(sub_lines) + "</sub>")
    top = (len(lines) * (TITLE_FONT_PX + 7)
           + len(sub_lines) * (SUBTITLE_FONT_PX + 4) + TITLE_PAD_PX + 12)

    return text, top


def _title_layout(text: str) -> dict:
    """Pin the title to the top of the figure rather than centring it in the top margin.

    Plotly's default centres the title in whatever margin it is given, so a title that grows by a
    line grows downwards into the plot -- which is where make_subplots puts the column labels.
    Anchoring it to the container top means the margin can be sized for the title plus a band for
    those labels, and neither moves when the other changes.
    """
    return dict(text=text, yref="container", y=1.0, yanchor="top", pad=dict(t=TITLE_PAD_PX))


def _axis_names(index: int) -> tuple[str, str]:
    suffix = "" if index == 1 else str(index)
    return f"x{suffix}", f"y{suffix}"


# ---------------------------------------------------------------------------
# (a) image grid: rows and columns are two correctors
# ---------------------------------------------------------------------------
def plot_scan_grid(data: pl.DataFrame,
                   row_knob: str,
                   col_knob: str,
                   display_size: int = GRID_DISPLAY_SIZE,
                   aggregate: str = AGGREGATE,
                   output_name: str | None = None,
                   show: bool = True,
                   roi: bool = False) -> go.Figure:
    """Lay the PCO images out on a grid spanned by two ELENA knobs.

    ``roi=True`` crops each cell to the MCP active area measured by fit_mcp_area.py, so the panels
    are all detector and none of the mount around it.
    """
    if aggregate not in AGGREGATES:
        # the cell code tests `aggregate == "mean"`, so anything unrecognised would quietly fall
        # through to "first" and show one repeat while the caller believed it was seeing all three
        raise ValueError(f"aggregate must be one of {AGGREGATES}, not {aggregate!r}")

    for knob in (row_knob, col_knob):
        if knob not in data.columns:
            raise KeyError(f"{knob} is not a column; use one of the requested settings "
                           f"{hd.ALL_SETPOINTS} or the readbacks {hd.ALL_READBACKS}")

    complete = data.filter(pl.col("has_image")
                           & pl.col(row_knob).is_not_null()
                           & pl.col(col_knob).is_not_null())
    # say which runs fell out rather than letting them vanish: a run whose Batman snapshot predates
    # the steering knobs has no knob columns at all, and silently missing cells look like real gaps
    dropped = sorted(set(data.get_column("Run Number").to_list())
                     - set(complete.get_column("Run Number").to_list()))
    if dropped:
        print(f"leaving {len(dropped)} run(s) out of the grid -- no image, or no "
              f"{row_knob} / {col_knob}: {dropped}")
    data = complete
    if data.is_empty():
        raise RuntimeError(f"no run has an image plus both {row_knob} and {col_knob}")

    data = data.with_columns(_binned(data, row_knob).alias("_row_value"),
                             _binned(data, col_knob).alias("_col_value"))

    # higher row values at the top, matching the convention in ELENA_beam_steering
    row_values = sorted(data.get_column("_row_value").unique().to_list(), reverse=True)
    col_values = sorted(data.get_column("_col_value").unique().to_list())

    cells: dict[tuple[float, float], list[dict]] = {}
    for row in data.iter_rows(named=True):
        cells.setdefault((row["_row_value"], row["_col_value"]), []).append(row)

    fig = make_subplots(
        rows=len(row_values), cols=len(col_values),
        # row values on the right; the column values go under the bottom row instead of on top, so
        # that each axis reads "name on the near side, values on the far side"
        row_titles=[f"{value:g}" for value in row_values],
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

            frames = [hd.get_image(match) for match in matches]
            if roi:
                frames = [hd.crop_to_mcp(frame) for frame in frames]
            images = [hd.downsample(frame, display_size, DOWNSAMPLE_HOW) for frame in frames]
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
                       hovertemplate=(f"{_label(row_knob)} = {matches[0]['_row_value']:g}"
                                      f"<br>{_label(col_knob)} = {matches[0]['_col_value']:g}"
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

    # Each axis is named on the near side and valued on the far side: the column knob's name above
    # the grid with its values under the bottom row, the row knob's name down the left with its
    # values on the right. Naming the axes here instead of in the title is what stops the title
    # from outgrowing the figure -- it used to carry both names and their readback channels.
    for grid_col, col_value in enumerate(col_values, start=1):
        fig.update_xaxes(title_text=f"{col_value:g}",
                         title_font=dict(size=AXIS_NAME_FONT_PX), title_standoff=6,
                         row=len(row_values), col=grid_col)

    # Each name is wrapped against the extent it is drawn along: the column name across the grid's
    # width, the row name along its height, since rotating it trades one for the other. A one-row
    # grid is only CELL_PX tall, which a name like "V_offset_requested [mm] (set, reads back on
    # V_offset_mm)" overruns by a factor of two if left on a single line.
    col_name = _wrap(_label(col_knob), CELL_PX * len(col_values), AXIS_NAME_FONT_PX)
    row_name = _wrap(_label(row_knob), CELL_PX * len(row_values), AXIS_NAME_FONT_PX)

    fig.add_annotation(text="<br>".join(col_name), showarrow=False,
                       font=dict(size=AXIS_NAME_FONT_PX), align="center",
                       xref="paper", yref="paper", x=0.5, y=1.0,
                       xanchor="center", yanchor="bottom", yshift=9)
    fig.add_annotation(text="<br>".join(row_name), showarrow=False,
                       font=dict(size=AXIS_NAME_FONT_PX), textangle=-90, align="center",
                       xref="paper", yref="paper", x=0.0, y=0.5,
                       xanchor="center", yanchor="middle",
                       xshift=-(20 + (len(row_name) - 1) * AXIS_NAME_LINE_PX // 2))

    empty = len(row_values) * len(col_values) - len(prepared)
    # exact requested setpoints need no binning; noisy readbacks do
    binned_axes = [name for name in (row_knob, col_knob) if name not in hd.ALL_SETPOINTS]
    binning_note = (", " + " and ".join(f"{name} binned to steps of {hd.bin_width_for(name):g}"
                                        for name in binned_axes)
                    if binned_axes else ", exact requested settings")
    width = CELL_PX * len(col_values) + 220
    # the axes now carry their own names, so the title only has to say what the figure is
    title, top_margin = _title(
        "PCO Edge scan",
        f"{data.height} runs, {len(prepared)} populated cells, {empty} empty{binning_note}",
        width, side_margins=180)
    top_margin += len(col_name) * AXIS_NAME_LINE_PX + 8
    fig.update_layout(
        title=_title_layout(title),
        height=CELL_PX * len(row_values) + 70 + top_margin,
        width=width,
        template="plotly_white",
        coloraxis=dict(colorscale=COLORSCALE, cmin=zmin, cmax=zmax,
                       colorbar=dict(title="signal", thickness=14)),
        margin=dict(t=top_margin, b=70,
                    l=36 + len(row_name) * AXIS_NAME_LINE_PX, r=90))

    os.makedirs(hd.PLOT_DIR, exist_ok=True)
    if output_name is None:
        output_name = f"pco_scan_{row_knob}_vs_{col_knob}.html"
    path = os.path.join(hd.PLOT_DIR, output_name)
    fig.write_html(path)
    print(f"wrote {path} ({os.path.getsize(path) / 1e6:.1f} MB)")

    if show:
        fig.show()

    return fig


# ---------------------------------------------------------------------------
# (b) knob matrix: every scanned knob on both axes
# ---------------------------------------------------------------------------
def plot_knob_matrix(data: pl.DataFrame,
                     knobs: list[str] | None = None,
                     observable: str = OBSERVABLE,
                     diagonal: str = DIAGONAL,
                     drop_constant: bool = DROP_CONSTANT,
                     output_name: str | None = None,
                     show: bool = True) -> go.Figure | None:
    """Pair plot of the driven knobs, coloured by an image observable.

    Returns None when nothing varied: a matrix of constants against constants says nothing, and at
    fourteen knobs it is a 196-cell figure saying it.
    """
    requested = knobs is not None
    if knobs is None:
        knobs = hd.ALL_SETPOINTS
    unknown = [name for name in knobs if name not in data.columns]
    if unknown and requested:
        raise KeyError(f"{unknown} are not columns; use the requested settings "
                       f"{hd.ALL_SETPOINTS} or the readbacks {hd.ALL_READBACKS}")
    knobs = [c for c in knobs if c in data.columns]
    if drop_constant:
        varying = _varying(data, knobs)
        if not varying:
            print("no knob varied across these runs; there is no matrix to draw")
            return None
        knobs = varying
    if not knobs:
        raise RuntimeError("no knob columns to plot")
    if observable not in data.columns:
        raise KeyError(f"{observable} is not a column of the data")

    data = data.filter(pl.col(observable).is_not_null())
    runs = data.get_column("Run Number").to_list()
    values = {knob: data.get_column(knob).to_list() for knob in knobs}
    observable_values = data.get_column(observable).to_list()

    size = len(knobs)
    fig = make_subplots(rows=size, cols=size,
                        horizontal_spacing=0.012, vertical_spacing=0.012,
                        shared_xaxes=True, shared_yaxes=False)

    for grid_row, y_knob in enumerate(knobs, start=1):
        for grid_col, x_knob in enumerate(knobs, start=1):
            on_diagonal = grid_row == grid_col

            if on_diagonal and diagonal == "histogram":
                fig.add_trace(go.Histogram(x=values[x_knob], marker_color="#666666",
                                           showlegend=False,
                                           hovertemplate=f"{x_knob}=%{{x}}<br>runs=%{{y}}"
                                                         "<extra></extra>"),
                              row=grid_row, col=grid_col)
            else:
                y_values = observable_values if on_diagonal else values[y_knob]
                y_name = observable if on_diagonal else y_knob
                fig.add_trace(
                    go.Scatter(x=values[x_knob], y=y_values, mode="markers",
                               customdata=list(zip(runs, observable_values)), showlegend=False,
                               marker=dict(size=7, coloraxis="coloraxis",
                                           color=observable_values,
                                           line=dict(width=0.5, color="#ffffff")),
                               hovertemplate=(f"{x_knob}=%{{x:.4g}}<br>{y_name}=%{{y:.4g}}"
                                              f"<br>{observable}=%{{customdata[1]:.4g}}"
                                              "<extra>run %{customdata[0]}</extra>")),
                    row=grid_row, col=grid_col)

            if grid_row == size:
                fig.update_xaxes(title_text=x_knob, title_font=dict(size=9),
                                 row=grid_row, col=grid_col)
            if grid_col == 1:
                fig.update_yaxes(title_text=y_knob, title_font=dict(size=9),
                                 row=grid_row, col=grid_col)
            fig.update_xaxes(showticklabels=False, ticks="", row=grid_row, col=grid_col)
            fig.update_yaxes(showticklabels=False, ticks="", row=grid_row, col=grid_col)

    diagonal_note = ("histogram of the sampled values" if diagonal == "histogram"
                     else f"knob vs {observable}")
    width = MATRIX_CELL_PX * size + 210
    title, top_margin = _title(
        f"ELENA knob scan matrix, {data.height} runs",
        f"marker colour = {observable}; diagonal = {diagonal_note}",
        width, side_margins=200)
    fig.update_layout(
        title=_title_layout(title),
        height=MATRIX_CELL_PX * size + 80 + top_margin,
        width=width,
        template="plotly_white",
        coloraxis=dict(colorscale="Viridis", colorbar=dict(title=observable, thickness=14)),
        margin=dict(t=top_margin, b=80, l=110, r=90))

    os.makedirs(hd.PLOT_DIR, exist_ok=True)
    if output_name is None:
        # the name predates the knobs being axes; kept so existing links to plots/ still resolve
        output_name = "pco_corrector_matrix.html"
    path = os.path.join(hd.PLOT_DIR, output_name)
    fig.write_html(path)
    print(f"wrote {path} ({os.path.getsize(path) / 1e6:.1f} MB)")

    if show:
        fig.show()

    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", default=None,
                        help="run numbers and/or first-last ranges (default: the RUNS constant)")
    parser.add_argument("--grid", nargs=2, metavar=("ROW", "COL"),
                        default=[ROW_KNOB, COL_KNOB],
                        help="knobs for the grid rows and columns (default: first two varying)")
    parser.add_argument("--axes", nargs="+", default=None,
                        help="columns on both axes of the matrix (default: every knob that varied). "
                             "Mixing steering knobs with corrector readbacks is the useful case, "
                             "e.g. --axes H_offset_requested V_offset_requested DHZE08R DVTE14T")
    parser.add_argument("--aggregate", choices=list(AGGREGATES), default=AGGREGATE,
                        help="how to combine repeats of one setting in a grid cell "
                             f"(default: {AGGREGATE})")
    parser.add_argument("--observable", default=OBSERVABLE, help="colour axis of the matrix")
    parser.add_argument("--diagonal", choices=["response", "histogram"], default=DIAGONAL)
    parser.add_argument("--skip-grid", action="store_true")
    parser.add_argument("--skip-matrix", action="store_true")
    parser.add_argument("--roi", action="store_true",
                        help="crop each grid cell to the MCP active area (see fit_mcp_area.py)")
    parser.add_argument("--force", action="store_true", help="re-download even if cached")
    parser.add_argument("--no-show", action="store_true", help="only write the html files")
    args = parser.parse_args()

    if not args.no_show:
        hd.use_browser_renderer()

    runs = hd.parse_runs(args.runs) if args.runs else RUNS
    data = hd.load_runs(runs, force=args.force)

    offenders = hd.check_mirror_channels(data)
    if offenders.is_empty():
        print("mirror check: every *L / *B / *N channel is the negative of its driven partner")
    else:
        print(f"mirror check: {offenders.height} channel readbacks are NOT mirrored -- the readback "
              f"was probably taken while the corrector was still ramping:")
        print(offenders)

    knob_state = hd.check_knob_state(data)
    if not knob_state.is_empty():
        print(f"knob check: {knob_state.height} logged knob state(s) disagree with the Batman "
              f"request -- the knobs are relative, so these runs carried state from earlier work:")
        print(knob_state)

    # ALL_SETPOINTS, not SETPOINTS: a run set steered by the knobs leaves every corrector setpoint
    # constant, and looking only at correctors makes such a scan register as no scan at all
    varying = _varying(data, hd.ALL_SETPOINTS)
    print(f"knobs that varied across these runs: {varying or 'none'}")
    varying_readbacks = _varying(data, hd.ALL_READBACKS)
    print(f"readbacks that varied across these runs: {varying_readbacks or 'none'}")

    if not args.skip_matrix:
        plot_knob_matrix(data, knobs=args.axes, observable=args.observable,
                         diagonal=args.diagonal, show=not args.no_show)

    if not args.skip_grid:
        row_knob, col_knob = args.grid
        if row_knob is None or col_knob is None:
            if len(varying) < 2:
                print(f"skipping the grid: it needs two knobs that varied, found {varying}. "
                      f"Tried the corrector setpoints {hd.SETPOINTS} and the steering knobs "
                      f"{hd.BEAM_SETPOINTS}.")
                raise SystemExit(0)
            row_knob, col_knob = varying[0], varying[1]
            print(f"grid knobs not given, using {row_knob} x {col_knob}")

        plot_scan_grid(data, row_knob, col_knob, aggregate=args.aggregate,
                       show=not args.no_show, roi=args.roi)
