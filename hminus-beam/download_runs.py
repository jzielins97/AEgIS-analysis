"""
Fill the local parquet cache from ALPACA. This is the only job this script does.

One run takes roughly 85 s through the ALPACA pipeline and one H- measurement takes about 90 s, so
downloading has to run alongside data-taking rather than after it. Start this in its own terminal
when you begin measuring and leave it running; the plotting scripts then read from the cache
instantly.

    python download_runs.py --runs 523357-523410           # catch up on a finished measurement
    python download_runs.py --watch --runs 523357-523410   # follow along while measuring
    python download_runs.py --watch --from 523411          # open ended, take whatever shows up
    python download_runs.py --backfill                     # widen cached runs to a new schema

--backfill exists because adding a column to the cache would otherwise mean re-downloading every
run. Asking ALPACA only for the non-camera variables makes speed_mode skip the PCO Edge tiff, so a
run gains its new columns in about 12 s instead of the 20-90 s a full download costs, and keeps the
image it already has. download_run() does this automatically when it meets a stale cache file.

Deliberately sequential -- do NOT add a --workers flag. ALPACA's finalize.generate() begins by
calling helpers.flush_directories(), which does shutil.rmtree() on the *shared*
python-analyses/data/processed_data directories. Two concurrent downloads would delete each other's
intermediates mid-flight, and load.check_for_datasets() walks that same directory and raises if
another process is writing to it.
"""

import argparse
import os
import time

import polars as pl

import hminus_data as hd

# hminus_data loads python-analyses/.env on import, which ALPACA needs before this import works
import ALPACA.applications.utils.utils as utils  # noqa: E402

POLL_SECONDS = 20
MEASUREMENT_SECONDS = 90.0   # rough H- cadence, only used to say whether we are keeping pace


def daq_run_list() -> list[int]:
    """Run numbers currently on the DAQ, or an empty list if it cannot be reached."""
    try:
        return utils.get_run_list_from_daq()
    except Exception as e:
        print(f"  could not reach the DAQ ({type(e).__name__}: {e})")
        return []


def is_cached(run: int) -> bool:
    """True when the run already has a cache file carrying the current schema."""
    path = hd.run_parquet_path(run)
    if not os.path.exists(path):
        return False

    columns = pl.read_parquet_schema(path).keys()
    return all(column in columns for column in hd.EXPECTED_COLUMNS)


def pending_runs(targets: list[int] | None,
                 available: list[int],
                 start_run: int | None,
                 hold_latest: bool,
                 force: bool = False) -> list[int]:
    """Runs that exist on the DAQ, are not cached yet, and are safe to download."""
    if targets is None:
        candidates = [run for run in available if start_run is None or run >= start_run]
    else:
        candidates = [run for run in targets if run in available]

    if hold_latest and available:
        # The newest run on the DAQ may still be being written. A run with a successor is finished,
        # so hold back the maximum until the next one appears.
        newest = max(available)
        candidates = [run for run in candidates if run < newest]

    if force:
        return candidates

    return [run for run in candidates if not is_cached(run)]


def download_batch(runs: list[int], force: bool, elapsed: list[float]) -> tuple[int, int]:
    """Download each run in turn, reporting progress. Returns (succeeded, failed)."""
    succeeded = failed = 0

    for position, run in enumerate(runs, start=1):
        started = time.time()
        try:
            data = hd.download_run(run, force=force)
        except Exception as e:
            failed += 1
            print(f"  run {run}  FAILED after {time.time() - started:.0f} s "
                  f"({type(e).__name__}: {e})")
            continue

        took = time.time() - started
        elapsed.append(took)
        succeeded += 1

        note = ""
        if not data.select("has_image").item():
            note = "  *** NO PCO EDGE IMAGE ***"

        average = sum(elapsed) / len(elapsed)
        print(f"  run {run}  ok  {took:5.1f} s   (avg {average:5.1f} s, "
              f"{len(runs) - position} left in this batch){note}")

    return succeeded, failed


def cached_runs() -> list[int]:
    """Every run number that has a cache file, whatever its schema."""
    runs = []
    for name in os.listdir(hd.DATA_DIR):
        stem, extension = os.path.splitext(name)
        if extension == ".parquet" and stem.isdigit():
            runs.append(int(stem))

    return sorted(runs)


def backfill(targets: list[int] | None) -> None:
    """Bring cached runs up to the current schema without re-reading their images."""
    available = set(cached_runs())
    runs = sorted(available) if targets is None else [r for r in targets if r in available]
    stale = [run for run in runs if hd.missing_columns(run)]

    if not stale:
        print(f"all {len(runs)} cached run(s) already carry the current schema")
        return

    print(f"{len(stale)} of {len(runs)} cached run(s) need columns added")
    started = time.time()
    done = needs_download = 0
    for run in stale:
        try:
            if hd.backfill_run(run):
                done += 1
            else:
                needs_download += 1
        except Exception as e:
            needs_download += 1
            print(f"  run {run}  FAILED ({type(e).__name__}: {e})")

    print(f"\nbackfilled {done} run(s) in {time.time() - started:.0f} s")
    if needs_download:
        print(f"{needs_download} run(s) need a full re-download (an image column is missing); "
              f"they will be fetched the next time they are loaded")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--runs", nargs="+",
                        help="run numbers and/or inclusive first-last ranges to download")
    source.add_argument("--from", dest="start_run", type=int,
                        help="download every run on the DAQ from this number onwards")
    parser.add_argument("--watch", action="store_true",
                        help="keep polling the DAQ for new runs instead of exiting when done")
    parser.add_argument("--poll", type=float, default=POLL_SECONDS,
                        help=f"seconds between DAQ checks in watch mode (default {POLL_SECONDS})")
    parser.add_argument("--include-latest", action="store_true",
                        help="also take the newest run on the DAQ, which may still be being written")
    parser.add_argument("--force", action="store_true",
                        help="re-download runs that are already cached")
    parser.add_argument("--backfill", action="store_true",
                        help="add columns missing from already cached runs, without re-reading "
                             "their images, then exit")
    args = parser.parse_args()

    if args.runs is None and args.start_run is None and not args.watch and not args.backfill:
        parser.error("give --runs, --from, --watch, or --backfill")

    targets = hd.parse_runs(args.runs) if args.runs else None

    if args.backfill:
        backfill(targets)
        return

    if targets:
        print(f"target: {len(targets)} runs, {min(targets)}-{max(targets)}")

    # A finished measurement needs no readiness guard; only hold the newest run back while following
    # along live.
    hold_latest = args.watch and not args.include_latest

    elapsed: list[float] = []
    total_ok = total_failed = 0
    started = time.time()

    try:
        while True:
            available = daq_run_list()
            runs = pending_runs(targets, available, args.start_run, hold_latest, args.force)

            if runs:
                print(f"{time.strftime('%H:%M:%S')}  {len(runs)} run(s) to download")
                ok, failed = download_batch(runs, args.force, elapsed)
                total_ok += ok
                total_failed += failed

            if not args.watch:
                if not runs:
                    print("nothing to download; everything requested is already cached")
                break

            # --force would otherwise re-download the same runs on every poll
            args.force = False

            if targets and all(is_cached(run) for run in targets):
                print("every requested run is cached, stopping")
                break

            if not runs:
                newest = max(available) if available else "?"
                print(f"{time.strftime('%H:%M:%S')}  waiting for new runs "
                      f"(newest on DAQ: {newest})", end="\r")
            time.sleep(args.poll)

    except KeyboardInterrupt:
        print("\ninterrupted")

    print(f"\ndownloaded {total_ok}, failed {total_failed}, "
          f"total {time.time() - started:.0f} s")
    if elapsed:
        average = sum(elapsed) / len(elapsed)
        pace = "keeping up with" if average < MEASUREMENT_SECONDS else "slower than"
        print(f"average {average:.1f} s per run -- {pace} the ~{MEASUREMENT_SECONDS:.0f} s "
              f"measurement cadence")


if __name__ == "__main__":
    main()
