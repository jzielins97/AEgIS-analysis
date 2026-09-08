"""
Work out how the ELENA steering knobs are applied to the corrector voltages, from a list of runs.

ELENA can be steered two ways: by setting corrector voltages directly, or by setting knobs -- an
offset (mm) and an angle (mrad) per plane. The knob path converts a knob change into correction
voltages and adds them to the correctors. This script recovers that conversion as a matrix::

    corrector_readback = ELENA_<n>_CORRECTOR_V + M @ [H_offset, V_offset, H_angle, V_angle]

The Batman corrector setpoints are the reference the correction is added to, so they are the
knob-zero steering: matrix plus reference is everything an absolute-knob implementation needs.

Only the Batman setpoints and the ELENA readbacks are used. The camera frame is never downloaded,
so runs whose MCP acquisition was empty work exactly as well as runs with a picture.

    python fit_beam_transform.py --runs 523415-523441
    python fit_beam_transform.py --runs 523415-523441 492712-492716   # add angle variation
    python fit_beam_transform.py --runs 523415-523441 --csv

A run set only determines the knobs it actually moved. A scan that holds the angles at zero cannot
say anything about the angle columns, and this script says so instead of reporting the zeros that
least squares would hand back.
"""

import argparse
import os

import numpy as np
import polars as pl

import hminus_data as hd

# A corrector counts as driven when the knobs move it by more than this across the run set's own
# knob range. Readback jitter is ~0.05 V, so 1 V is comfortably above the noise while still far
# below the hundreds of volts a real response produces.
ATOL = 1.0

# How far ELENA's logged knob state may differ from what Batman requested before the run is thrown
# out. The two agree to ~5e-5 on a healthy run.
KNOB_ATOL = 0.01

# Below this, a singular value of the mean-centred design counts as zero and the fit is rank
# deficient -- some combination of knobs was never varied independently.
RANK_TOL = 1e-8


def usable_runs(data: pl.DataFrame, knob_atol: float = KNOB_ATOL) -> pl.DataFrame:
    """Drop runs whose logged knob state disagrees with the request, explaining each one.

    The knobs are relative, so a run can start with state left over from earlier work; such a run
    was not taken at the knob values it recorded and would drag the fit badly. Run 523328 is the
    example in the current cache -- including it moves the worst residual from 0.15 V to 247 V.
    """
    offenders = hd.check_knob_state(data, atol=knob_atol)
    if offenders.is_empty():
        return data

    bad = sorted(set(offenders.get_column("Run Number").to_list()))
    print(f"leaving out {len(bad)} run(s) whose logged knob state disagrees with the Batman "
          f"request -- stale relative-knob state, or steered by setting voltages directly:")
    for row in offenders.iter_rows(named=True):
        print(f"    run {row['Run Number']:.0f}  {row['knob']:<20s} "
              f"requested {row['requested']:>9.4g}, logged {row['logged']:>9.4g}")

    return data.filter(~pl.col("Run Number").is_in(bad))


def identifiable_knobs(data: pl.DataFrame,
                       knobs: list[str],
                       rank_tol: float = RANK_TOL) -> tuple[list[str], np.ndarray]:
    """The knobs this run set can actually determine, plus the design's singular values.

    Least squares happily returns a minimum-norm answer for a knob that never moved, and zeros for
    an unvaried column look exactly like a physical "this knob does nothing". Dropping such columns
    before the fit is what keeps the two apart.
    """
    varying = [knob for knob in knobs
               if knob in data.columns and float(data.get_column(knob).std() or 0.0) > 0.0]
    if not varying:
        return [], np.array([])

    design = np.column_stack([data.get_column(knob).to_numpy() for knob in varying])
    singular = np.linalg.svd(design - design.mean(axis=0), compute_uv=False)

    return varying, singular


def fit_transform(data: pl.DataFrame, knobs: list[str]) -> dict:
    """Least-squares fit of ``readback - setpoint`` against ``knobs``, per corrector.

    Referencing each corrector to its own Batman setpoint rather than fitting the raw readback is
    what lets runs from different campaigns, taken at different base steering, go into one fit.
    """
    design = np.column_stack([data.get_column(knob).to_numpy() for knob in knobs]
                             + [np.ones(data.height)])

    results = {}
    for device in hd.SET_CORRECTORS:
        setpoint = hd.DEVICE_REQUESTED.get(device)
        if device not in data.columns or setpoint not in data.columns:
            continue

        # A run missing this corrector's readback or setpoint is dropped for this corrector only.
        # Whole channels used to be skipped instead, which threw away every corrector as soon as one
        # run in the set had a gap -- 523326 and 523327 have no ELENA readbacks at all.
        readback = data.get_column(device)
        usable = (readback.is_not_null() & data.get_column(setpoint).is_not_null()).to_numpy()
        if usable.sum() < len(knobs) + 2:
            continue

        rows = design[usable]
        # a knob that is constant among the surviving rows cannot be fitted from them
        if np.linalg.matrix_rank(rows - rows.mean(axis=0), tol=RANK_TOL) < len(knobs):
            continue

        target = (readback - data.get_column(setpoint)).to_numpy()[usable]
        coefficients, *_ = np.linalg.lstsq(rows, target, rcond=None)
        residual = target - rows @ coefficients

        # what the knobs did to this corrector over the range the run set actually covered; a
        # coefficient alone cannot say whether that is a real response or rounding noise
        swing = sum(abs(coefficient) * float(np.ptp(rows[:, index]))
                    for index, coefficient in enumerate(coefficients[:-1]))

        results[device] = {"coefficients": coefficients[:-1],
                           "intercept": float(coefficients[-1]),
                           "max_residual": float(np.abs(residual).max()),
                           "swing": swing,
                           "runs": int(usable.sum())}

    return results


def report(data: pl.DataFrame, knobs: list[str], singular: np.ndarray, results: dict,
           atol: float = ATOL) -> None:
    """Print the matrix, the knob-zero reference and everything that qualifies them."""
    unit = {knob: hd.KNOB_UNIT.get(knob, "?") for knob in knobs}
    header = "".join(f"{knob.replace('_requested', ''):>16s}" for knob in knobs)
    units = "".join(f"{'[V/' + unit[knob] + ']':>16s}" for knob in knobs)

    print(f"\ncorrector_readback = <its Batman setpoint> + M @ "
          f"[{', '.join(knob.replace('_requested', '') for knob in knobs)}]\n")
    print(f"{'corrector':<10s}{'':<6s}{header}{'max resid':>12s}")
    print(f"{'':<16s}{units}{'[V]':>12s}")

    driven, quiet = [], []
    for device, fit in results.items():
        (driven if fit["swing"] > atol else quiet).append(device)

    for device in driven:
        fit = results[device]
        short = hd.DEVICE_CORRECTOR.get(device, "")
        coefficients = "".join(f"{value:>16.3e}" for value in fit["coefficients"])
        note = "" if fit["runs"] == data.height else f"   ({fit['runs']} of {data.height} runs)"
        print(f"{device:<10s}{short.split('_')[0]:<6s}{coefficients}"
              f"{fit['max_residual']:>12.3f}{note}")

    if quiet:
        print(f"\nnot driven by these knobs (moved < {atol:g} V over the run set's knob range):")
        print(f"    {', '.join(quiet)}")

    print(f"\nknob-zero reference -- the steering these knobs correct away from:")
    for device in hd.SET_CORRECTORS:
        setpoint = hd.DEVICE_REQUESTED.get(device)
        if setpoint not in data.columns:
            continue
        values = data.get_column(setpoint).drop_nulls().unique().sort().to_list()
        shown = f"{values[0]:.6g}" if len(values) == 1 else f"{len(values)} values {values}"
        print(f"    {setpoint:<22s} ({device}) {shown}")

    print(f"\ndesign singular values: {np.round(singular, 3)}")


def write_csv(knobs: list[str], results: dict, path: str) -> None:
    """Save the matrix as one row per corrector, so it can be reused without refitting."""
    rows = [{"corrector": device,
             **{knob: float(value) for knob, value in zip(knobs, fit["coefficients"])},
             "intercept": fit["intercept"], "max_residual": fit["max_residual"]}
            for device, fit in results.items()]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pl.DataFrame(rows).write_csv(path)
    print(f"wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", required=True,
                        help="run numbers and/or inclusive first-last ranges, e.g. 523415-523441")
    parser.add_argument("--atol", type=float, default=ATOL,
                        help="a corrector counts as driven above this many volts of movement")
    parser.add_argument("--knob-atol", type=float, default=KNOB_ATOL,
                        help="how far the logged knob state may differ from the request")
    parser.add_argument("--csv", action="store_true", help="also write the matrix into plots/")
    parser.add_argument("--force", action="store_true", help="re-download even if cached")
    args = parser.parse_args()

    runs = hd.parse_runs(args.runs)
    data = hd.load_settings(runs, force=args.force)
    print(f"loaded settings for {data.height} run(s), no camera frames read")

    data = usable_runs(data, knob_atol=args.knob_atol)
    if data.height < 2:
        raise SystemExit(f"need at least 2 usable runs to fit, have {data.height}")

    knobs, singular = identifiable_knobs(data, hd.BEAM_SETPOINTS)
    unvaried = [knob for knob in hd.BEAM_SETPOINTS if knob not in knobs]
    if unvaried:
        print(f"\nthese knobs did not move in this run set, so it cannot determine their effect: "
              f"{', '.join(unvaried)}")
        print("    add runs that scanned them; least squares would otherwise report a bare 0 here, "
              "which is indistinguishable from a real 'this knob does nothing'")
    if not knobs:
        raise SystemExit("no knob varied across these runs; there is nothing to fit")

    if np.linalg.matrix_rank(np.diag(singular), tol=RANK_TOL) < len(knobs):
        raise SystemExit(f"the knobs {knobs} were not varied independently (singular values "
                         f"{np.round(singular, 4)}); the fit cannot separate them")

    results = fit_transform(data, knobs)
    if not results:
        raise SystemExit("no corrector has both a readback and a setpoint across these runs")

    print(f"fitting {len(results)} corrector(s) against {len(knobs)} knob(s) "
          f"over {data.height} run(s)")
    report(data, knobs, singular, results, atol=args.atol)

    if args.csv:
        write_csv(knobs, results, os.path.join(hd.PLOT_DIR, "beam_transform.csv"))


if __name__ == "__main__":
    main()
