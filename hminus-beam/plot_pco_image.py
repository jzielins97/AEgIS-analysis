"""
Open one PCO Edge image in its own window, with x and y profiles and the ELENA settings beside it.

Reads the local parquet cache (see hminus_data.py) and downloads the run through ALPACA if it is
not cached yet -- run download_runs.py first if you want that to be instant.

    python plot_pco_image.py 523357
    python plot_pco_image.py 523357 523369 --contrast 5
    python plot_pco_image.py 523357-523410 --no-show     # whole scan, PNGs instead of windows
    python plot_pco_image.py 523369-523371 --average     # the three repeats of one scan point

One window per run, all opened at once, so past a dozen runs use --no-show and look at the PNGs it
writes into plots/.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

import hminus_data as hd

COLORMAP = "magma"
CONTRAST = 1.0           # vmax = peak / CONTRAST; raise it to bring out faint structure
FIGSIZE = (13.5, 7.5)

# how the text panel is grouped, label -> list of columns
BEAM_KEYS = ["H_offset_mm", "V_offset_mm", "H_angle_mrad", "V_angle_mrad",
             "catch_delay", "beam_stopper_position"]
IMAGE_KEYS = ["PCO_signal_sum", "PCO_max_signal_in_px", "PCO_mean_background", "PCO_std_background"]

# requested-value column for each beam readback, so the panel can show "readback (set X)"
BEAM_REQUESTED = {"H_offset_mm": "H_offset_requested", "V_offset_mm": "V_offset_requested",
                  "H_angle_mrad": "H_angle_requested", "V_angle_mrad": "V_angle_requested"}


def _value(row: dict, key: str) -> str:
    value = row.get(key)
    if value is None:
        return "n/a"
    if isinstance(value, str):
        return value
    return f"{value:.6g}"


def info_text(row: dict, profiles: dict, image: np.ndarray,
              header: list[str] | None = None) -> str:
    """The ELENA settings block shown next to the image."""
    if header is None:
        header = [f"run {row['Run Number']}", _value(row, "PCO_acq_time")]
    lines = list(header) + [""]

    lines.append("REQUESTED SETPOINTS")
    for name in hd.SETPOINTS:
        if row.get(name) is not None:
            lines.append(f"  {name:<22s} {row[name]:>10.6g}")
    lines.append("")

    lines.append("BEAM")
    for key in BEAM_KEYS:
        text = f"  {key:<22s} {_value(row, key):>10s}"
        requested = row.get(BEAM_REQUESTED.get(key, ""))
        if requested is not None:
            text += f"  (set {requested:g})"
        lines.append(text)
    lines.append("")

    lines.append("CORRECTORS   readback (set)")
    for device in hd.SET_CORRECTORS:
        if row.get(device) is None:
            continue
        requested = row.get(hd.DEVICE_REQUESTED.get(device, ""))
        suffix = f"  (set {requested:g})" if requested is not None else ""
        lines.append(f"  {device:<10s} {hd.DEVICE_CORRECTOR[device]:<18s}"
                     f" {row[device]:>10.6g}{suffix}")
    lines.append("")

    lines.append("IMAGE")
    for key in IMAGE_KEYS:
        lines.append(f"  {key:<22s} {_value(row, key):>10s}")

    # PCO_max_signal_in_px comes from the full 5120x5120 frame, so a single hot pixel there is 25x
    # brighter than in the 5x5 coarsened image on screen. Show both so the colour scale makes sense.
    shown_max = float(image.max())
    lines.append(f"  {'(that max is full res)':<22s}")
    lines.append(f"  {'max in shown image':<22s} {shown_max:>10.6g}")

    # Not a beam/no-beam test: a single hot pixel gives a high ratio on an otherwise empty frame.
    # The sign of PCO_signal_sum above is the reliable indicator -- it integrates the whole image,
    # so it goes strongly positive with beam and sits near zero or negative without.
    background = row.get("PCO_std_background")
    if background:
        lines.append(f"  {'max / background std':<22s} {shown_max / background:>10.1f}")

    lines.append(f"  {'peak row, column':<22s} "
                 f"{profiles['slice_row']:>5d},{profiles['slice_column']:>5d}")

    return "\n".join(lines)


def _draw_profile(axis, position, projection, slice_values, vertical: bool) -> None:
    """Draw a projection and a centre slice on one axis, each normalised to its own maximum.

    The projection sums over ~1000 rows while the slice is a single one, so they differ by about
    three orders of magnitude. Normalising each to its own peak is what makes the overlay
    comparable; the true peaks go in the legend.
    """
    projection_peak = float(np.max(np.abs(projection))) or 1.0
    slice_peak = float(np.max(np.abs(slice_values))) or 1.0

    curves = [(projection / projection_peak, f"projection (peak {projection_peak:.4g})",
               dict(color="#1f77b4", lw=1.2)),
              (slice_values / slice_peak, f"slice (peak {slice_peak:.4g})",
               dict(color="#d62728", lw=0.9, ls="--", alpha=0.65))]

    for values, label, style in curves:
        if vertical:
            axis.plot(values, position, label=label, **style)
        else:
            axis.plot(position, values, label=label, **style)

    axis.grid(alpha=0.25)
    axis.legend(fontsize=6, loc="upper right", framealpha=0.7)


def plot_image(row: dict,
               contrast: float = CONTRAST,
               save: bool = False,
               image: np.ndarray | None = None,
               label: str | None = None,
               header: list[str] | None = None,
               save_name: str | None = None) -> plt.Figure:
    """One window: image, both marginal profiles, and the ELENA settings.

    Pass ``image`` to show something other than this row's own frame -- the averaged view uses it.
    """
    if image is None:
        image = hd.get_image(row)
    if image is None:
        raise ValueError(f"run {row['Run Number']} has no PCO Edge image "
                         f"(has_image={row.get('has_image')})")

    label = label or f"run {row['Run Number']}"
    profiles = hd.image_profiles(image)
    height, width = image.shape

    fig = plt.figure(figsize=FIGSIZE)
    grid = fig.add_gridspec(2, 3, width_ratios=[4, 1.2, 2.6], height_ratios=[4, 1.2],
                            hspace=0.12, wspace=0.06,
                            left=0.05, right=0.985, top=0.94, bottom=0.08)

    ax_image = fig.add_subplot(grid[0, 0])
    ax_y = fig.add_subplot(grid[0, 1], sharey=ax_image)
    ax_x = fig.add_subplot(grid[1, 0], sharex=ax_image)
    ax_text = fig.add_subplot(grid[:, 2])
    ax_colorbar = fig.add_subplot(grid[1, 1])

    peak = float(image.max())
    mesh = ax_image.imshow(image, cmap=COLORMAP, origin="upper", aspect="equal",
                           vmin=0.0, vmax=peak / contrast if contrast else peak,
                           interpolation="nearest")
    ax_image.set_title(f"{label}   PCO Edge {width}x{height} (5x5 coarsened)", fontsize=10)
    ax_image.set_ylabel("row [px]")
    ax_image.tick_params(labelbottom=False)

    # mark where the slices were taken
    ax_image.axhline(profiles["slice_row"], color="w", lw=0.5, alpha=0.4)
    ax_image.axvline(profiles["slice_column"], color="w", lw=0.5, alpha=0.4)

    _draw_profile(ax_x, np.arange(width), profiles["x_projection"], profiles["x_slice"],
                  vertical=False)
    ax_x.set_xlabel("column [px]")
    ax_x.set_ylabel("normalised")

    _draw_profile(ax_y, np.arange(height), profiles["y_projection"], profiles["y_slice"],
                  vertical=True)
    ax_y.set_xlabel("normalised")
    ax_y.tick_params(labelleft=False)

    # a slim bar inside the cell, so the y-profile's tick labels above it stay readable
    ax_colorbar.axis("off")
    fig.colorbar(mesh, cax=ax_colorbar.inset_axes([0.05, 0.42, 0.9, 0.13]),
                 orientation="horizontal", label="signal")

    ax_text.axis("off")
    ax_text.text(0.0, 1.0, info_text(row, profiles, image, header), transform=ax_text.transAxes,
                 va="top", ha="left", family="monospace", fontsize=7.5)

    if hasattr(fig.canvas, "manager") and fig.canvas.manager is not None:
        try:
            fig.canvas.manager.set_window_title(label)
        except Exception:
            pass

    if save:
        os.makedirs(hd.PLOT_DIR, exist_ok=True)
        path = os.path.join(hd.PLOT_DIR, save_name or f"pco_image_{row['Run Number']}.png")
        fig.savefig(path, dpi=150)
        print(f"wrote {path}")

    return fig


def plot_average(runs: list[int],
                 contrast: float = CONTRAST,
                 save: bool = False,
                 force: bool = False,
                 show: bool = True) -> plt.Figure | None:
    """Average several repeats of one setting into a single figure.

    Refuses when the runs were not taken at the same settings: averaging images from different
    optics is meaningless, so the mismatch is reported instead of silently producing a blend.
    """
    data = hd.load_runs(runs, force=force)

    skipped = data.filter(~pl.col("has_image")).get_column("Run Number").to_list()
    if skipped:
        print(f"no PCO Edge image for runs {skipped}, leaving them out of the average")
    data = data.filter(pl.col("has_image"))
    if data.is_empty():
        print("none of these runs has a PCO Edge image")
        return None

    present = data.get_column("Run Number").to_list()
    differing = hd.differing_settings(data)
    if differing:
        print(f"refusing to average: these {len(present)} runs were not taken at the same settings")
        for key, values in differing.items():
            spread = ", ".join(f"{run}={value!r}" for run, value in zip(present, values))
            print(f"  {key}: {spread}")
        print("average a single scan point at a time, e.g. the three repeats 523369-523371")
        return None

    mean_image, averaged = hd.average_runs(data)
    print(f"settings match across runs {present}, averaging {len(present)} image(s)")

    label = f"mean of {len(present)} runs {present[0]}-{present[-1]}"
    header = [label, ", ".join(str(run) for run in present),
              "settings verified identical; scalars below are means"]

    figure = plot_image(averaged, contrast=contrast, save=save, image=mean_image, label=label,
                        header=header,
                        save_name=f"pco_image_avg_{present[0]}-{present[-1]}.png")

    if show:
        plt.show()
    else:
        plt.close(figure)

    return figure


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


# opening more windows than this at once is unmanageable, so warn instead of doing it silently
MANY_WINDOWS = 12


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+",
                        help="run numbers and/or inclusive first-last ranges, e.g. 523357-523410")
    parser.add_argument("--contrast", type=float, default=CONTRAST,
                        help="vmax = peak / contrast; raise it to bring out faint structure")
    parser.add_argument("--save", action="store_true", help="also write a PNG into plots/")
    parser.add_argument("--no-show", action="store_true",
                        help="write the PNGs without opening windows (implies --save)")
    parser.add_argument("--force", action="store_true", help="re-download even if cached")
    parser.add_argument("--average", action="store_true",
                        help="check the runs share the same ELENA settings, then show one figure "
                             "of their averaged image")
    args = parser.parse_args()

    runs = _parse_runs(args.runs)
    save = args.save or args.no_show

    if args.average:
        plot_average(runs, contrast=args.contrast, save=save, force=args.force,
                     show=not args.no_show)
        return

    if not args.no_show and len(runs) > MANY_WINDOWS:
        print(f"{len(runs)} runs would open {len(runs)} windows at once; "
              f"use --no-show to write PNGs into plots/ instead")
        return

    shown = 0
    for run in runs:
        try:
            data = hd.download_run(run, force=args.force)
            figure = plot_image(data.row(0, named=True), contrast=args.contrast, save=save)
            shown += 1
            if args.no_show:
                # otherwise every figure stays open and matplotlib complains past 20 of them
                plt.close(figure)
        except Exception as e:
            print(f"run {run}: {type(e).__name__}: {e}")

    if not shown:
        print("nothing to show")
    elif args.no_show:
        print(f"wrote {shown} figure(s) into {hd.PLOT_DIR}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
