"""
Data layer for the H- beam measurements imaged on the PCO Edge camera.

Runs are pulled through ALPACA once and cached as one parquet per run in ``data/<run>.parquet``.
Every plotting script in this directory reads from that cache, so the (slow) ALPACA pipeline runs
at most once per run.

The cached image is the gold pipeline's ``coarsed_image``: the raw PCO Edge frame is 5120x5120
uint16 (~210 MB per run as float64), while the 5x5 block mean is 1024x1024 and comes out of
CMOSDataAnalysis for free. See ALPACA/data/pipelines/gold/CMOSDataAnalysis.py.
"""

import importlib.util
import logging
import os

import numpy as np
import polars as pl
from dotenv import load_dotenv

logging.basicConfig(format="%(levelname)s:%(name)s:%(message)s")
_log = logging.getLogger(__name__)
_log.setLevel(logging.INFO)


def _load_alpaca_dotenv() -> None:
    """Load python-analyses/.env before ALPACA is imported.

    ALPACA calls a bare ``load_dotenv()``, which only searches upwards from the current working
    directory, so running a script from this folder leaves DB_CONN_PER_SOCKET unset and
    PostgresDB_utils raises at import time. Locate the .env next to the installed ALPACA package
    instead, so these scripts work from any working directory.
    """
    spec = importlib.util.find_spec("ALPACA")
    if spec is None or not spec.submodule_search_locations:
        return

    dotenv = os.path.join(os.path.dirname(spec.submodule_search_locations[0]), ".env")
    if os.path.exists(dotenv):
        load_dotenv(dotenv)
    else:
        _log.warning("no .env found at %s; ALPACA downloads will likely fail", dotenv)


_load_alpaca_dotenv()

import ALPACA.data.finalize as finalize  # noqa: E402  (must follow _load_alpaca_dotenv)

DETECTOR = "PCOEdge"
ACQ = "acq_0"

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
PLOT_DIR = os.path.join(os.path.dirname(__file__), "plots")

# ---------------------------------------------------------------------------
# ELENA correctors
# ---------------------------------------------------------------------------
# ALPACA short name -> CERN device name. Mirrors valid_name_to_variable in
# ALPACA/data/pipelines/silver/ElenaReadOutsCleaning.py.
CORRECTOR_DEVICE = {
    "H1_Corrector_R": "DHZE05R", "H1_Corrector_L": "DHZE05L",
    "V1_Corrector_T": "DVTE05T", "V1_Corrector_B": "DVTE05B",
    "H2_Corrector_R": "DHZE08R", "H2_Corrector_L": "DHZE08L",
    "V2_Corrector_T": "DVTE08T", "V2_Corrector_B": "DVTE08B",
    "H3_Corrector_R": "DHZE14R", "H3_Corrector_L": "DHZE14L",
    "V3_Corrector_T": "DVTE14T", "V3_Corrector_B": "DVTE14B",
    "QD1_P": "QDNE08P", "QD1_N": "QDNE08N",
    "QF1_P": "QFNE09P", "QF1_N": "QFNE09N",
    "QD2_P": "QDNE14P", "QD2_N": "QDNE14N",
    "QF2_P": "QFNE15P", "QF2_N": "QFNE15N",
}
DEVICE_CORRECTOR = {device: short for short, device in CORRECTOR_DEVICE.items()}

# Only these channels are actually driven by the control system; *L / *B / *N mirror them with the
# opposite sign. Same 10 knobs the steering optimiser uses (ELENA_beam_steering/optimiser_status.py).
SET_CORRECTORS = ["DHZE05R", "DVTE05T", "DHZE08R", "DVTE08T", "DHZE14R", "DVTE14T",
                  "QDNE08P", "QFNE09P", "QDNE14P", "QFNE15P"]

# set channel -> the channel that should read back as its negative
MIRROR_CHANNEL = {"DHZE05R": "DHZE05L", "DVTE05T": "DVTE05B",
                  "DHZE08R": "DHZE08L", "DVTE08T": "DVTE08B",
                  "DHZE14R": "DHZE14L", "DVTE14T": "DVTE14B",
                  "QDNE08P": "QDNE08N", "QFNE09P": "QFNE09N",
                  "QDNE14P": "QDNE14N", "QFNE15P": "QFNE15N"}

# ---------------------------------------------------------------------------
# variables requested from ALPACA -> column name in the parquet
# ---------------------------------------------------------------------------
_IMAGE_VARIABLES = {
    f"{DETECTOR}*{ACQ}*background_corrected*coarsed_image": "PCO_img",
    f"{DETECTOR}*{ACQ}*background_corrected*signal_sum": "PCO_signal_sum",
    f"{DETECTOR}*{ACQ}*background_corrected*signal_density_per_mm2": "PCO_signal_density_per_mm2",
    f"{DETECTOR}*{ACQ}*background_corrected*max_signal_in_px": "PCO_max_signal_in_px",
    f"{DETECTOR}*{ACQ}*background_corrected*mean_background": "PCO_mean_background",
    f"{DETECTOR}*{ACQ}*background_corrected*std_background": "PCO_std_background",
    f"{DETECTOR}*{ACQ}*background_corrected*x_center_of_mass": "PCO_x_center_of_mass",
    f"{DETECTOR}*{ACQ}*background_corrected*y_center_of_mass": "PCO_y_center_of_mass",
    f"{DETECTOR}*{ACQ}*acq_time_str": "PCO_acq_time",
}

_ELENA_BEAM_VARIABLES = {
    "ELENA_Parameters*H_offset_mm": "H_offset_mm",
    "ELENA_Parameters*V_offset_mm": "V_offset_mm",
    "ELENA_Parameters*H_angle_mrad": "H_angle_mrad",
    "ELENA_Parameters*V_angle_mrad": "V_angle_mrad",
    "ELENA_Parameters*delay": "catch_delay",
    "ELENA_Parameters*beam_stopper_position": "beam_stopper_position",
    # 'ELENA_Parameters*elena_gate_valve_state' is deliberately not requested. ElenaReadOutsCleaning
    # matches the bare substring 'value' against every line, and each corrector readout line ends in
    # 'MEAS.V.VALUE#value', so the last one (DSHE20R) overwrites the real 'LNE.VVGBH.0226/State#value
    # Open'. The observable therefore reports a corrector current instead of the valve state.
}

_ELENA_CORRECTOR_VARIABLES = {
    f"ELENA_Parameters*{short}": device for short, device in CORRECTOR_DEVICE.items()
}

# What the control system was *asked* to set, as opposed to the ELENA_Parameters readbacks above.
# Batman carries a full snapshot of the control-system parameters (334 keys for a typical run), not
# just the knobs named on the artiq command line, so every run has all ten setpoints regardless of
# which ones its scan varied. Use varying_setpoints() to find the ones a given run set actually
# moved.
#
# Each driven corrector has exactly one setpoint here, which is what makes *R / *T / *P the driven
# channels: e.g. ELENA_H1_CORRECTOR_V = 166.0 against a DHZE05R readback of 166.0511. Scanning on
# the setpoint is more robust than binning the readback, because it is the exact requested number.
REQUESTED_TO_DEVICE = {
    "ELENA_H1_CORRECTOR_V": "DHZE05R", "ELENA_V1_CORRECTOR_V": "DVTE05T",
    "ELENA_H2_CORRECTOR_V": "DHZE08R", "ELENA_V2_CORRECTOR_V": "DVTE08T",
    "ELENA_H3_CORRECTOR_V": "DHZE14R", "ELENA_V3_CORRECTOR_V": "DVTE14T",
    "ELENA_QD1_V": "QDNE08P", "ELENA_QF1_V": "QFNE09P",
    "ELENA_QD2_V": "QDNE14P", "ELENA_QF2_V": "QFNE15P",
}
DEVICE_REQUESTED = {device: requested for requested, device in REQUESTED_TO_DEVICE.items()}
SETPOINTS = list(REQUESTED_TO_DEVICE)

_BATMAN_VARIABLES = {f"Batman*{ACQ}*{name}": name for name in SETPOINTS}
_BATMAN_VARIABLES.update({
    f"Batman*{ACQ}*ELENA_H_OFFSET": "H_offset_requested",
    f"Batman*{ACQ}*ELENA_V_OFFSET": "V_offset_requested",
    f"Batman*{ACQ}*ELENA_H_ANGLE": "H_angle_requested",
    f"Batman*{ACQ}*ELENA_V_ANGLE": "V_angle_requested",
})

# variable -> column, in the order they are written to the parquet
VARIABLE_TO_COLUMN = {"Run_Number*Run_Number*__value": "Run Number",
                      **_IMAGE_VARIABLES,
                      **_BATMAN_VARIABLES,
                      **_ELENA_BEAM_VARIABLES,
                      **_ELENA_CORRECTOR_VARIABLES}

# columns every cache file must carry; a file missing any of them predates a schema change and is
# re-downloaded on access (see download_run)
EXPECTED_COLUMNS = list(VARIABLE_TO_COLUMN.values()) + ["PCO_img_height", "PCO_img_width",
                                                        "has_image"]

# The variables that do not come from the camera. Asking ALPACA for only these makes
# bronze.remove_files_in_speed_mode drop the PCO Edge tiff from the file list, so the pipeline never
# reads or flattens the 26 M pixel frame and CMOSDataAnalysis finds nothing to analyse. That is
# where nearly all the per-run time goes, which is what makes backfill_run cheap.
SCALAR_VARIABLE_TO_COLUMN = {variable: column for variable, column in VARIABLE_TO_COLUMN.items()
                             if not variable.startswith(f"{DETECTOR}*")}
IMAGE_COLUMNS = {column for variable, column in VARIABLE_TO_COLUMN.items()
                 if variable.startswith(f"{DETECTOR}*")}

# the ELENA scalars shown in a picture's description, in display order. The driven correctors are
# named by device; describe_elena appends their requested setpoint automatically.
DESCRIPTION_KEYS = ["H_offset_mm", "V_offset_mm", "H_angle_mrad", "V_angle_mrad",
                    "catch_delay", "beam_stopper_position"] + SET_CORRECTORS


def run_parquet_path(run: int) -> str:
    """Path of the local cache file for a single run."""
    return os.path.join(DATA_DIR, f"{run}.parquet")


def _is_missing(value) -> bool:
    """ALPACA reports an unavailable observable as a bare float nan rather than an array."""
    if value is None:
        return True
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return True
    return False


def download_run(run: int,
                 force: bool = False,
                 directories_to_flush: list | None = None,
                 speed_mode: bool = True,
                 verbosing: bool = False) -> pl.DataFrame:
    """Return a one-row DataFrame for ``run``, downloading it through ALPACA if not cached.

    The result is cached in ``data/<run>.parquet``. Pass ``force=True`` to re-download.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    path = run_parquet_path(run)

    if os.path.exists(path) and not force:
        missing = missing_columns(run)
        if not missing:
            _log.info("run %s: cache hit (%s)", run, path)
            return pl.read_parquet(path)

        # Only scalar columns are missing? Then fetch just those, which skips the camera frame and
        # takes seconds instead of a minute.
        if backfill_run(run, directories_to_flush=directories_to_flush, verbosing=verbosing):
            return pl.read_parquet(path)

        _log.info("run %s: cached file predates a schema change (missing %s), re-downloading",
                  run, ", ".join(missing))

    if directories_to_flush is None:
        directories_to_flush = ["bronze", "gold", "datasets", "elog"]

    _log.info("run %s: downloading from ALPACA", run)
    raw = finalize.generate(
        first_run=run,
        last_run=run,
        elog_results_filename=str(run),
        known_bad_runs=[],
        # generate() appends 'Batman' to the list it is handed, so give it a throwaway copy
        variables_of_interest=list(VARIABLE_TO_COLUMN.keys()),
        directories_to_flush=directories_to_flush,
        speed_mode=speed_mode,
        verbosing=verbosing,
        # coarsed_image is not a database column, so the DB shortcut could never satisfy this
        # request; forcing local processing also avoids the interactive Tk prompts.
        force_local_processing=True,
    )

    row = {}
    image = None
    for variable, column in VARIABLE_TO_COLUMN.items():
        try:
            value = raw[variable.replace("*", "_")][0]
        except (KeyError, IndexError):
            value = None

        if _is_missing(value):
            if column != "PCO_img":
                row[column] = None
            continue

        if column == "PCO_img":
            image = np.asarray(value, dtype=np.float32)
        else:
            row[column] = value.item() if isinstance(value, np.generic) else value

    if row.get("Run Number") is None:
        row["Run Number"] = run
    row["Run Number"] = int(row["Run Number"])

    if image is not None and image.ndim != 2:
        _log.warning("run %s: unexpected image shape %s, dropping it", run, image.shape)
        image = None

    if image is None:
        _log.warning("run %s: no PCO Edge image", run)
        row["PCO_img_height"] = None
        row["PCO_img_width"] = None
        row["has_image"] = False
        data = pl.DataFrame(row).with_columns(
            pl.lit(None, dtype=pl.List(pl.Float32)).alias("PCO_img"))
    else:
        row["PCO_img_height"] = int(image.shape[0])
        row["PCO_img_width"] = int(image.shape[1])
        row["has_image"] = True
        data = pl.DataFrame(row).with_columns(
            pl.Series("PCO_img", [image.ravel()], dtype=pl.List(pl.Float32)))

    data = data.with_columns(pl.col("Run Number").cast(pl.Int64),
                             pl.col("PCO_img_height").cast(pl.Int32),
                             pl.col("PCO_img_width").cast(pl.Int32))
    data.write_parquet(path, compression="zstd")
    _log.info("run %s: cached %.1f MB to %s", run, os.path.getsize(path) / 1e6, path)

    return data


def missing_columns(run: int) -> list[str] | None:
    """Columns a cached run lacks, [] if it is current, or None if it is not cached at all."""
    path = run_parquet_path(run)
    if not os.path.exists(path):
        return None

    columns = pl.read_parquet_schema(path).keys()

    return [column for column in EXPECTED_COLUMNS if column not in columns]


def backfill_run(run: int,
                 directories_to_flush: list | None = None,
                 verbosing: bool = False) -> bool:
    """Add missing non-image columns to an already cached run, without re-reading its image.

    Returns True when the file was updated. Falls back to returning False when the run is not
    cached, is already current, or is missing an image column (which needs a full download).
    """
    missing = missing_columns(run)
    if missing is None:
        _log.info("run %s: not cached, nothing to backfill", run)
        return False
    if not missing:
        return False
    if IMAGE_COLUMNS.intersection(missing):
        _log.info("run %s: missing image column(s) %s, needs a full download",
                  run, ", ".join(sorted(IMAGE_COLUMNS.intersection(missing))))
        return False

    if directories_to_flush is None:
        directories_to_flush = ["bronze", "gold", "datasets", "elog"]

    _log.info("run %s: backfilling %s", run, ", ".join(missing))
    raw = finalize.generate(
        first_run=run,
        last_run=run,
        elog_results_filename=str(run),
        known_bad_runs=[],
        variables_of_interest=list(SCALAR_VARIABLE_TO_COLUMN.keys()),
        directories_to_flush=directories_to_flush,
        speed_mode=True,
        verbosing=verbosing,
        force_local_processing=True,
    )

    column_to_variable = {column: variable
                          for variable, column in SCALAR_VARIABLE_TO_COLUMN.items()}
    additions = {}
    for column in missing:
        value = None
        variable = column_to_variable.get(column)
        if variable is not None:
            try:
                value = raw[variable.replace("*", "_")][0]
            except (KeyError, IndexError):
                value = None
            if _is_missing(value):
                value = None
            elif isinstance(value, np.generic):
                value = value.item()
        additions[column] = value

    data = pl.read_parquet(run_parquet_path(run))
    data = data.with_columns([pl.lit(value).alias(column) for column, value in additions.items()])
    data.write_parquet(run_parquet_path(run), compression="zstd")
    _log.info("run %s: backfilled %d column(s)", run, len(additions))

    return True


def load_runs(runs: list[int], force: bool = False, **kwargs) -> pl.DataFrame:
    """Load (downloading if needed) every run in ``runs`` into a single DataFrame."""
    frames = []
    for run in sorted(runs):
        try:
            frames.append(download_run(run, force=force, **kwargs))
        except Exception as e:  # a single bad run must not kill the whole set
            _log.error("run %s failed: %s: %s", run, type(e).__name__, e)

    if not frames:
        raise RuntimeError(f"none of the requested runs could be loaded: {runs}")

    return pl.concat(frames, how="diagonal_relaxed").sort("Run Number")


def get_image(row: dict | pl.DataFrame) -> np.ndarray | None:
    """Rebuild the 2D image of a single run from its flattened cache column."""
    if isinstance(row, pl.DataFrame):
        row = row.row(0, named=True)

    flat = row.get("PCO_img")
    height, width = row.get("PCO_img_height"), row.get("PCO_img_width")
    if flat is None or height is None or width is None:
        return None

    return np.reshape(np.asarray(flat, dtype=np.float32), (height, width))


def downsample(image: np.ndarray, max_size: int, how: str = "mean") -> np.ndarray:
    """Block-reduce an image so neither side exceeds ``max_size``.

    Rows and columns that do not fit a whole block are cropped, so the reduction is exact.

    ``how="mean"`` preserves the integrated signal and is the right choice for anything
    quantitative. ``how="max"`` preserves peaks instead: the beam spot on the PCO Edge covers very
    few pixels, so block-averaging for display dilutes it into the background. Use "max" when the
    picture is only being looked at, "mean" when the numbers matter.
    """
    if image is None or max_size is None or max(image.shape) <= max_size:
        return image

    factor = int(np.ceil(max(image.shape) / max_size))
    height = (image.shape[0] // factor) * factor
    width = (image.shape[1] // factor) * factor
    blocks = image[:height, :width].reshape(height // factor, factor, width // factor, factor)

    if how == "max":
        return blocks.max(axis=(1, 3))
    if how == "mean":
        return blocks.mean(axis=(1, 3))

    raise ValueError(f"unknown reduction {how!r}, expected 'mean' or 'max'")


def setpoints(row: dict | pl.DataFrame) -> dict:
    """Every requested ELENA setpoint of a run, e.g. {'ELENA_QD1_V': 1000.0, ...}."""
    if isinstance(row, pl.DataFrame):
        row = row.row(0, named=True)

    return {name: row[name] for name in SETPOINTS if row.get(name) is not None}


def image_profiles(image: np.ndarray,
                   slice_row: int | None = None,
                   slice_column: int | None = None) -> dict:
    """Projections and centre slices of an image, for the marginal plots of the viewer.

    The projections sum over a whole axis, so they use all the signal and stay smooth when the beam
    is faint. The slices are a single row/column, taken by default through the peak of each
    projection. The peak is used rather than the cached PCO_x/y_center_of_mass because ALPACA
    computes those on the original 5120x5120 frame while this image is the 1024x1024 coarsened one.
    """
    x_projection = image.sum(axis=0)   # one value per column
    y_projection = image.sum(axis=1)   # one value per row

    if slice_row is None:
        slice_row = int(np.argmax(y_projection))
    if slice_column is None:
        slice_column = int(np.argmax(x_projection))

    slice_row = int(np.clip(slice_row, 0, image.shape[0] - 1))
    slice_column = int(np.clip(slice_column, 0, image.shape[1] - 1))

    return {"x_projection": x_projection, "y_projection": y_projection,
            "x_slice": image[slice_row, :], "y_slice": image[:, slice_column],
            "slice_row": slice_row, "slice_column": slice_column}


def _format_value(value) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (int, float, np.number)) and not isinstance(value, bool):
        return f"{value:.4g}"
    return str(value)


def describe_elena(row: dict | pl.DataFrame,
                   keys: list[str] | None = None,
                   sep: str = "<br>") -> str:
    """Render the ELENA settings of a run as a description string for a plot."""
    if isinstance(row, pl.DataFrame):
        row = row.row(0, named=True)
    if keys is None:
        keys = DESCRIPTION_KEYS

    lines = []
    for key in keys:
        if key not in row:
            continue
        if key in DEVICE_CORRECTOR:
            label = f"{key} ({DEVICE_CORRECTOR[key]})"
            text = _format_value(row[key])
            # for a driven corrector, show what was asked for next to what came back
            requested = row.get(DEVICE_REQUESTED.get(key))
            if requested is not None:
                text += f" (set {_format_value(requested)})"
        else:
            label, text = key, _format_value(row[key])
        lines.append(f"{label} = {text}")

    return sep.join(lines)


def check_mirror_channels(data: pl.DataFrame, atol: float = 1.0) -> pl.DataFrame:
    """Report runs where a *L / *B / *N channel is not the negative of its driven partner.

    Returns one row per (run, channel pair) that fails, so an empty frame means every mirrored
    corrector behaved as expected.
    """
    offenders = []
    for run_row in data.iter_rows(named=True):
        for driven, mirror in MIRROR_CHANNEL.items():
            set_value, mirror_value = run_row.get(driven), run_row.get(mirror)
            if set_value is None or mirror_value is None:
                continue
            deviation = abs(set_value + mirror_value)
            if deviation > atol:
                offenders.append({"Run Number": run_row["Run Number"],
                                  "driven": driven, "mirror": mirror,
                                  "driven_value": set_value, "mirror_value": mirror_value,
                                  "abs(driven + mirror)": deviation})

    if not offenders:
        return pl.DataFrame(schema={"Run Number": pl.Int64, "driven": pl.Utf8, "mirror": pl.Utf8,
                                    "driven_value": pl.Float64, "mirror_value": pl.Float64,
                                    "abs(driven + mirror)": pl.Float64})

    return pl.DataFrame(offenders)


# Corrector readbacks jitter by ~0.01 between runs while real scan steps are tens to hundreds, so
# group settings into bins of this width rather than comparing floats or rounding to decimals.
BIN_WIDTH = 1.0


def bin_values(values: pl.Series, bin_width: float = BIN_WIDTH) -> pl.Series:
    """Snap corrector readbacks onto a grid of ``bin_width``, so repeats of one setting collapse."""
    return (values / bin_width).round() * bin_width


# The parameters that must agree before averaging several runs together. These are the requested
# numbers, so they compare exactly -- unlike the readbacks, which jitter by ~0.01 between runs.
COMPARE_KEYS = SETPOINTS + ["H_offset_requested", "V_offset_requested",
                            "H_angle_requested", "V_angle_requested", "catch_delay"]


def differing_settings(data: pl.DataFrame, keys: list[str] | None = None) -> dict[str, list]:
    """Which of ``keys`` are not the same across every run, mapped to the values they took."""
    if keys is None:
        keys = COMPARE_KEYS

    differing = {}
    for key in keys:
        if key not in data.columns:
            continue
        values = data.get_column(key).to_list()
        if len({repr(value) for value in values}) > 1:
            differing[key] = values

    return differing


def average_runs(data: pl.DataFrame) -> tuple[np.ndarray, dict]:
    """Mean image and mean scalars over every run in ``data`` that has an image.

    Numeric columns are averaged, so the readbacks and image observables shown alongside the mean
    image describe the same set of runs. Text columns keep the first run's value.
    """
    rows = [row for row in data.iter_rows(named=True) if row.get("has_image")]
    if not rows:
        raise ValueError("none of these runs has a PCO Edge image")

    images = [get_image(row) for row in rows]
    shapes = {image.shape for image in images}
    if len(shapes) > 1:
        raise ValueError(f"cannot average images of differing shapes: {sorted(shapes)}")

    mean_image = np.mean(images, axis=0, dtype=np.float64).astype(np.float32)

    averaged = {}
    for column in rows[0]:
        if column == "PCO_img":
            continue
        values = [row[column] for row in rows if row[column] is not None]
        if values and all(isinstance(value, (int, float)) and not isinstance(value, bool)
                          for value in values):
            averaged[column] = float(np.mean(values))
        else:
            averaged[column] = values[0] if values else None

    averaged["Run Number"] = [row["Run Number"] for row in rows]
    averaged["PCO_img_height"], averaged["PCO_img_width"] = mean_image.shape
    averaged["has_image"] = True

    return mean_image, averaged


def varying_setpoints(data: pl.DataFrame, candidates: list[str] | None = None) -> list[str]:
    """The requested setpoints that a run set actually scanned.

    Every run carries all ten setpoints, so this is how you tell which knobs a measurement moved.
    No binning is needed: these are the exact requested numbers, not noisy readbacks.
    """
    if candidates is None:
        candidates = SETPOINTS

    return [name for name in candidates
            if name in data.columns and data.get_column(name).drop_nulls().n_unique() > 1]


def varying_correctors(data: pl.DataFrame,
                       correctors: list[str] | None = None,
                       bin_width: float = BIN_WIDTH) -> list[str]:
    """The subset of ``correctors`` that actually took more than one value across ``data``."""
    if correctors is None:
        correctors = SET_CORRECTORS

    varying = []
    for corrector in correctors:
        if corrector not in data.columns:
            continue
        values = bin_values(data.get_column(corrector).drop_nulls(), bin_width).unique()
        if len(values) > 1:
            varying.append(corrector)

    return varying
