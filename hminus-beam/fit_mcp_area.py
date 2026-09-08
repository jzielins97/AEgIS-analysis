"""
Measure the MCP's active area from the LED picture of the detector and write it down once.

Run 523443 is the detector lit by its LED with no beam, so the active area shows up as a bright
disc inside the dark mount. Fitting that circle gives the exact MCP centre to refer beam positions
to, and a region of interest that keeps the rest of the analysis (and the pictures it draws) inside
the part of the frame that can actually see anything.

    python fit_mcp_area.py                       # fit run 523443, write data/mcp_roi.json
    python fit_mcp_area.py --dry-run             # fit and draw, write nothing
    python fit_mcp_area.py --vmax 30             # set the colour ceiling of the diagnostic figure
    python fit_mcp_area.py --level 12            # set the threshold that separates disc from mount
    python fit_mcp_area.py --center 614,597 --radius 275   # skip the search, refine from here
    python fit_mcp_area.py --pick                # click points on the rim yourself

The result goes to data/mcp_roi.json; hminus_data.load_mcp_roi and crop_to_mcp read it from there.
Always look at plots/mcp_roi_<run>.png before trusting it.
"""

import argparse
import datetime
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from scipy import ndimage as ndi

import hminus_data as hd

RUN = 523443             # the LED picture of the MCP
COLORMAP = "magma"

# The LED floods the whole frame, so the disc only separates from its surroundings once the top of
# the scale is clipped: on 523443 the peak is 2957 while the disc itself sits around 20-40.
VMAX_PERCENTILE = 88.0

SMOOTH_PX = 5            # median filter width; kills hot pixels without moving an edge
MIN_AREA_FRACTION = 0.03  # a disc candidate must cover at least this much of the frame
RAYS = 720               # one rim sample every half degree
SEARCH_LOW, SEARCH_HIGH = 0.75, 1.30   # radius window searched for the rim, in units of the seed
TRIM_SIGMA = 2.5         # rim points further than this from the current circle are dropped
TRIM_PASSES = 6

# past these the fit is not to be trusted without a look at the figure
MAX_RMS_PX = 3.0
MIN_KEPT_FRACTION = 0.4


# ---------------------------------------------------------------------------
# preparing the picture
# ---------------------------------------------------------------------------
def destripe_rows(image: np.ndarray) -> np.ndarray:
    """Undo the per-row offset ALPACA leaves in a PCO Edge frame.

    For this camera the gold pipeline "background correction" subtracts, from every row, the mean
    of that row over full-res columns 0-150 (CMOSDataAnalysis.analyse_background). On a beam run
    that strip really is background. With the LED on it is lit, so each row loses a different,
    unknown amount and the frame gains horizontal banding.

    The offset varies smoothly from row to row, so it can be recovered from the median difference
    between neighbouring rows -- the median ignores the minority of columns where the picture
    genuinely changes down the frame. Integrating those differences gives the offset up to one
    constant, which is fixed by putting the darkest part of the frame at zero.
    """
    steps = np.median(image[1:, :] - image[:-1, :], axis=1)
    offsets = np.concatenate([[0.0], np.cumsum(steps)])
    corrected = image - offsets[:, None]

    return corrected - np.percentile(corrected, 2)


def prepare(image: np.ndarray) -> np.ndarray:
    """Destripe, then median filter so single hot pixels cannot pass for an edge."""
    return ndi.median_filter(destripe_rows(image), SMOOTH_PX)


# ---------------------------------------------------------------------------
# finding the disc
# ---------------------------------------------------------------------------
def _roundness(component: np.ndarray) -> tuple[float, float, float, float]:
    """(centre row, centre column, mean boundary radius, relative spread) of a binary blob.

    The relative spread is the scatter of centre-to-boundary distances divided by their mean: 0 for
    a perfect circle, and large for anything else. It is what separates the MCP from the other
    bright patches in the frame.
    """
    rows, columns = np.nonzero(component)
    center_row, center_col = rows.mean(), columns.mean()

    boundary = component ^ ndi.binary_erosion(component)
    boundary_rows, boundary_cols = np.nonzero(boundary)
    distances = np.hypot(boundary_rows - center_row, boundary_cols - center_col)

    return center_row, center_col, distances.mean(), distances.std() / distances.mean()


def disc_candidates(image: np.ndarray, level: float) -> list[dict]:
    """Bright blobs enclosed by darkness at this threshold, described by how round they are.

    The MCP sits in a dark mount, so the disc is a *hole* in the dark part of the picture. Looking
    for holes rather than for bright blobs is what keeps the flood-lit background -- which is
    brighter still, but open to the frame edge -- from swamping the search.
    """
    dark = image < level
    holes = ndi.binary_fill_holes(dark) & ~dark

    labels, count = ndi.label(holes)
    if count == 0:
        return []

    minimum_area = MIN_AREA_FRACTION * image.size
    found = []
    for index, area in enumerate(ndi.sum(holes, labels, range(1, count + 1)), start=1):
        if area < minimum_area:
            continue
        component = labels == index
        center_row, center_col, radius, spread = _roundness(component)
        found.append({"level": float(level), "area": float(area), "spread": float(spread),
                      "center_row": float(center_row), "center_col": float(center_col),
                      # the area is the honest size estimate; the boundary mean is dragged inwards
                      # by any bite taken out of the blob
                      "radius": float(np.sqrt(area / np.pi)), "boundary_radius": float(radius)})

    return found


def seed_circle(image: np.ndarray, level: float | None = None) -> dict:
    """A first guess at the disc: the roundest large hole, over a scan of thresholds.

    Scanning rather than fixing a threshold means the LED brightness does not have to be known in
    advance; ``--level`` pins it when the scan picks something silly.
    """
    if level is not None:
        levels = [level]
    else:
        levels = np.percentile(image, np.arange(20, 71, 5))

    candidates = [candidate for value in levels for candidate in disc_candidates(image, value)]
    if not candidates:
        raise RuntimeError(
            "no large round hole found in the dark parts of the frame. Look at the picture "
            "(python plot_pco_image.py <run> --contrast 100) and rerun with --level, "
            "--center/--radius, or --pick.")

    return min(candidates, key=lambda candidate: candidate["spread"])


# ---------------------------------------------------------------------------
# the rim and the circle through it
# ---------------------------------------------------------------------------
def rim_points(image: np.ndarray,
               center_row: float,
               center_col: float,
               radius: float,
               rays: int = RAYS,
               step: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Sub-pixel rim positions, one per ray cast outwards from the seed centre.

    The rim is taken as the steepest fall in brightness along each ray rather than a fixed
    brightness, because the LED lights the disc very unevenly: on 523443 the left of the disc is
    four times brighter than the right, so any single level would cut the rim in the wrong place on
    one side. A gradient extremum does not care how bright either side is.
    """
    angles = np.linspace(0, 2 * np.pi, rays, endpoint=False)
    radii = np.arange(SEARCH_LOW * radius, SEARCH_HIGH * radius, step)

    sample_rows = center_row + radii[None, :] * np.sin(angles[:, None])
    sample_cols = center_col + radii[None, :] * np.cos(angles[:, None])
    profiles = ndi.map_coordinates(image, [sample_rows.ravel(), sample_cols.ravel()],
                                   order=1, mode="nearest").reshape(rays, -1)
    profiles = ndi.gaussian_filter1d(profiles, 3.0, axis=1)

    slope = np.gradient(profiles, step, axis=1)
    peak = np.argmin(slope, axis=1)                     # most negative = brightness falling outwards
    inside = (peak > 0) & (peak < slope.shape[1] - 1)

    # parabola through the three samples around the extremum, for the sub-pixel position
    index = np.arange(rays)
    before = slope[index, np.clip(peak - 1, 0, None)]
    at = slope[index, peak]
    after = slope[index, np.clip(peak + 1, None, slope.shape[1] - 1)]
    curvature = before - 2 * at + after
    shift = np.where(np.abs(curvature) > 1e-9,
                     0.5 * (before - after) / np.where(curvature == 0, 1.0, curvature), 0.0)

    found = radii[peak] + np.clip(shift, -1.0, 1.0) * step

    return (center_row + found[inside] * np.sin(angles[inside]),
            center_col + found[inside] * np.cos(angles[inside]))


def fit_circle(rows: np.ndarray, columns: np.ndarray) -> tuple[float, float, float]:
    """Least-squares circle through a set of points (Kasa's algebraic fit)."""
    design = np.c_[columns, rows, np.ones(len(rows))]
    target = columns ** 2 + rows ** 2
    solution, *_ = np.linalg.lstsq(design, target, rcond=None)

    center_col, center_row = solution[0] / 2, solution[1] / 2
    radius = np.sqrt(solution[2] + center_col ** 2 + center_row ** 2)

    return center_row, center_col, radius


def fit_circle_robustly(rows: np.ndarray, columns: np.ndarray) -> dict:
    """Circle fit that throws away the rim points the mount and its shadows contribute.

    A plain least-squares fit is dragged off the disc by the few rays that find a stronger edge on
    something else -- the bright stripe near column 110, the shadow across the bottom. Refitting
    without the points furthest from the current circle, a few times over, converges on the rim
    itself.
    """
    keep = np.ones(len(rows), dtype=bool)
    center_row = center_col = radius = 0.0

    for _ in range(TRIM_PASSES):
        center_row, center_col, radius = fit_circle(rows[keep], columns[keep])
        residuals = np.hypot(rows - center_row, columns - center_col) - radius
        spread = float(np.std(residuals[keep]))
        if spread == 0:
            break
        proposed = np.abs(residuals) < TRIM_SIGMA * spread
        if proposed.sum() < 3 or np.array_equal(proposed, keep):
            break
        keep = proposed
    else:
        # ran out of passes with a trim still pending, so fit the set that was actually kept
        center_row, center_col, radius = fit_circle(rows[keep], columns[keep])

    residuals = np.hypot(rows - center_row, columns - center_col) - radius

    return {"center_row": float(center_row), "center_col": float(center_col),
            "radius": float(radius), "keep": keep,
            "rms_px": float(np.sqrt(np.mean(residuals[keep] ** 2))),
            "n_points": int(len(rows)), "n_kept": int(keep.sum())}


# ---------------------------------------------------------------------------
# picking the rim by hand
# ---------------------------------------------------------------------------
def pick_rim(image: np.ndarray, vmax: float) -> tuple[np.ndarray, np.ndarray]:
    """Open a window and collect rim points clicked by the user.

    Left click adds a point, right click removes the last, middle click or Enter finishes.
    """
    figure, axis = plt.subplots(figsize=(9, 9))
    axis.imshow(image, cmap=COLORMAP, origin="upper", vmin=0.0, vmax=vmax,
                interpolation="nearest")
    axis.set_title("click at least 3 points around the rim of the MCP\n"
                   "right click undoes, middle click or Enter finishes", fontsize=10)
    figure.tight_layout()

    clicked = figure.ginput(n=-1, timeout=0, show_clicks=True)
    plt.close(figure)

    if len(clicked) < 3:
        raise RuntimeError(f"need at least 3 points on the rim, got {len(clicked)}")

    columns = np.array([point[0] for point in clicked])
    rows = np.array([point[1] for point in clicked])

    return rows, columns


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------
def roi_record(run: int, fit: dict, extra: dict) -> dict:
    """Everything a downstream script could want about the circle, in one flat dictionary."""
    radius_mm = fit["radius"] * hd.COARSE_PX_TO_MM

    return {"run": run,
            "fitted": datetime.datetime.now().isoformat(timespec="seconds"),
            "coarsen": hd.COARSEN,
            "px_to_mm": hd.PX_TO_MM,
            "coarse_px_to_mm": hd.COARSE_PX_TO_MM,
            # cached (5x5 coarsened) frame, which is what every plotting script handles
            "center_row_px": fit["center_row"],
            "center_col_px": fit["center_col"],
            "radius_px": fit["radius"],
            # the raw 5120x5120 sensor, for anything that goes back to ALPACA
            "center_row_raw_px": fit["center_row"] * hd.COARSEN,
            "center_col_raw_px": fit["center_col"] * hd.COARSEN,
            "radius_raw_px": fit["radius"] * hd.COARSEN,
            "radius_mm": radius_mm,
            "diameter_mm": 2 * radius_mm,
            "fit": {"n_points": fit["n_points"], "n_kept": fit["n_kept"],
                    "rms_px": fit["rms_px"], **extra}}


def draw_diagnostics(image: np.ndarray,
                     rows: np.ndarray,
                     columns: np.ndarray,
                     fit: dict,
                     record: dict,
                     vmax: float,
                     run: int,
                     save: bool = True) -> plt.Figure:
    """Three panels that make the fit checkable: the frame, the rim, and the radial profile."""
    center_row, center_col, radius = fit["center_row"], fit["center_col"], fit["radius"]
    keep = fit["keep"]

    figure, axes = plt.subplots(1, 3, figsize=(16.5, 5.8))

    axes[0].imshow(image, cmap=COLORMAP, origin="upper", vmin=0.0, vmax=vmax,
                   interpolation="nearest")
    axes[0].add_patch(Circle((center_col, center_row), radius, fill=False, color="#00ff88", lw=1.2))
    axes[0].plot(center_col, center_row, "+", color="#00ff88", ms=12, mew=1.5)
    # the dashed box is what hd.crop_to_mcp will keep once this fit is saved. Derived from the fit
    # in hand rather than from hd.mcp_roi_box, which would still be reading the previous one.
    reach = radius + hd.ROI_MARGIN_PX
    left, top = center_col - reach, center_row - reach
    axes[0].add_patch(plt.Rectangle((left, top), 2 * reach, 2 * reach,
                                    fill=False, color="#00ff88", ls="--", lw=0.8, alpha=0.7))
    axes[0].set_title(f"run {run}, destriped, vmax={vmax:.4g}", fontsize=10)
    axes[0].set_xlabel("column [px]")
    axes[0].set_ylabel("row [px]")

    margin = 0.25 * radius
    axes[1].imshow(image, cmap=COLORMAP, origin="upper", vmin=0.0, vmax=vmax,
                   interpolation="nearest")
    axes[1].plot(columns[keep], rows[keep], ".", color="#00ff88", ms=2, label=f"used ({keep.sum()})")
    axes[1].plot(columns[~keep], rows[~keep], ".", color="#ff3355", ms=3,
                 label=f"rejected ({(~keep).sum()})")
    axes[1].add_patch(Circle((center_col, center_row), radius, fill=False, color="w", lw=0.8,
                             alpha=0.6))
    axes[1].set_xlim(center_col - radius - margin, center_col + radius + margin)
    axes[1].set_ylim(center_row + radius + margin, center_row - radius - margin)
    axes[1].legend(fontsize=8, loc="upper right", framealpha=0.7)
    axes[1].set_title(f"rim points, residual rms {fit['rms_px']:.2f} px", fontsize=10)
    axes[1].set_xlabel("column [px]")

    # azimuthal median: robust to the bright stripe and to the shadow across the bottom
    grid_rows, grid_cols = np.mgrid[0:image.shape[0], 0:image.shape[1]]
    distance = np.hypot(grid_rows - center_row, grid_cols - center_col)
    edge = min(radius * 1.6, distance.max())
    bins = np.arange(0, edge, 2.0)
    which = np.digitize(distance.ravel(), bins) - 1
    values = image.ravel()
    profile = np.array([np.median(values[which == index]) if np.any(which == index) else np.nan
                        for index in range(len(bins) - 1)])

    axes[2].plot(bins[:-1] * hd.COARSE_PX_TO_MM, profile, color="#1f77b4", lw=1.2)
    axes[2].axvline(radius * hd.COARSE_PX_TO_MM, color="#ff3355", ls="--", lw=1.0,
                    label=f"fitted r = {record['radius_mm']:.2f} mm")
    axes[2].set_xlabel("distance from fitted centre [mm]")
    axes[2].set_ylabel("median signal")
    axes[2].set_ylim(0, np.nanmax(profile) * 1.15)
    axes[2].grid(alpha=0.25)
    axes[2].legend(fontsize=8)
    axes[2].set_title(f"active area {record['diameter_mm']:.2f} mm across", fontsize=10)

    figure.suptitle(f"MCP active area from run {run}: centre "
                    f"(row {center_row:.1f}, col {center_col:.1f}) px, "
                    f"radius {radius:.1f} px = {record['radius_mm']:.2f} mm", fontsize=11)
    figure.tight_layout(rect=(0, 0, 1, 0.95))

    if save:
        os.makedirs(hd.PLOT_DIR, exist_ok=True)
        path = os.path.join(hd.PLOT_DIR, f"mcp_roi_{run}.png")
        figure.savefig(path, dpi=140)
        print(f"wrote {path}")

    return figure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", nargs="?", type=int, default=RUN,
                        help=f"run holding the LED picture of the MCP (default {RUN})")
    parser.add_argument("--vmax", type=float,
                        help="colour ceiling of the diagnostic figure; the disc is invisible "
                             "without one because the LED floods the frame")
    parser.add_argument("--level", type=float,
                        help="brightness that separates the disc from the mount around it; "
                             "by default a scan of thresholds picks the roundest candidate")
    parser.add_argument("--center", help="seed centre as row,col, skipping the search")
    parser.add_argument("--radius", type=float, help="seed radius in pixels, skipping the search")
    parser.add_argument("--pick", action="store_true",
                        help="click points on the rim instead of searching for it")
    parser.add_argument("--dry-run", action="store_true",
                        help="fit and draw, but leave data/mcp_roi.json alone")
    parser.add_argument("--no-show", action="store_true", help="write the PNG without a window")
    parser.add_argument("--force", action="store_true", help="re-download the run")
    args = parser.parse_args()

    data = hd.download_run(args.run, force=args.force)
    image = hd.get_image(data)
    if image is None:
        raise SystemExit(f"run {args.run} has no PCO Edge image")

    prepared = prepare(image)
    vmax = args.vmax if args.vmax is not None else float(np.percentile(prepared, VMAX_PERCENTILE))

    if args.pick:
        rows, columns = pick_rim(prepared, vmax)
        seed = {"level": None, "spread": None, "source": "picked"}
    else:
        if args.center is not None or args.radius is not None:
            if args.center is None or args.radius is None:
                raise SystemExit("--center and --radius go together")
            center_row, center_col = (float(value) for value in args.center.split(","))
            seed = {"center_row": center_row, "center_col": center_col,
                    "radius": args.radius, "level": args.level, "spread": None,
                    "source": "seeded"}
        else:
            seed = seed_circle(prepared, args.level)
            seed["source"] = "auto"
            print(f"seed: centre (row {seed['center_row']:.1f}, col {seed['center_col']:.1f}), "
                  f"radius {seed['radius']:.1f} px, from level {seed['level']:.2f} "
                  f"(roundness spread {seed['spread']:.3f})")

        rows, columns = rim_points(prepared, seed["center_row"], seed["center_col"],
                                   seed["radius"])

    fit = fit_circle_robustly(rows, columns)
    record = roi_record(args.run, fit,
                        {"method": seed["source"], "vmax": vmax, "level": seed.get("level")})

    print(f"centre  row {fit['center_row']:8.2f} px   col {fit['center_col']:8.2f} px")
    print(f"radius      {fit['radius']:8.2f} px   = {record['radius_mm']:.3f} mm  "
          f"(diameter {record['diameter_mm']:.3f} mm)")
    print(f"residual rms {fit['rms_px']:.2f} px over {fit['n_kept']} of {fit['n_points']} "
          f"rim points")

    kept_fraction = fit["n_kept"] / max(fit["n_points"], 1)
    if fit["rms_px"] > MAX_RMS_PX or kept_fraction < MIN_KEPT_FRACTION:
        print("\n*** this fit does not look like a circle. Check the figure, then rerun with "
              "--level, --center/--radius, or --pick ***")

    figure = draw_diagnostics(prepared, rows, columns, fit, record, vmax, args.run)

    if args.dry_run:
        print("--dry-run: not writing " + hd.MCP_ROI_PATH)
    else:
        os.makedirs(hd.DATA_DIR, exist_ok=True)
        with open(hd.MCP_ROI_PATH, "w") as handle:
            json.dump(record, handle, indent=2)
        print(f"wrote {hd.MCP_ROI_PATH}")

    if args.no_show:
        plt.close(figure)
    else:
        plt.show()


if __name__ == "__main__":
    main()
