"""
Run fable_extract.py on every image in a folder.

BENCHMARK ONLY. Synthetic schedules only - never Dayforce data.

Put this file in the same folder as fable_extract.py.

Run:
    python run_benchmark.py images
    python run_benchmark.py images --week-start 2026-11-09
    python run_benchmark.py images --repeats 3 --max-cost 5

    images         folder containing the schedule images
    --week-start   week start date applied to every image (optional)
    --repeats      runs per image, for checking consistency (NFR-8). Default 1
    --max-cost     stop once estimated spend reaches this many USD. Default 5
    --out          folder for the results. Default "results"

Already-finished runs are skipped, so you can stop and restart safely.
"""

import argparse
import csv
import json
import time
from datetime import date
from pathlib import Path

from extract import (MEDIA_TYPES, MODEL, PRICE_IN_PER_MTOK,
                           PRICE_OUT_PER_MTOK, extract, to_dayforce)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--week-start", default=None)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--max-cost", type=float, default=5.0)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    week_start = date.fromisoformat(args.week_start) if args.week_start else None
    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)

    images = sorted(p for p in Path(args.folder).rglob("*")
                if p.suffix.lower() in MEDIA_TYPES)
    
    if not images:
        print(f"no images found in {args.folder}")
        return

    total_runs = len(images) * args.repeats
    print(f"{len(images)} images x {args.repeats} repeats = {total_runs} runs with {MODEL}")
    print(f"budget limit: ${args.max_cost:.2f}\n")

    summary_path = out_dir / "summary.csv"
    new_file = not summary_path.exists()
    spent = 0.0

    with open(summary_path, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["image", "run", "status", "shifts", "ready_for_dayforce",
                             "low_conf_fields", "input_tokens", "output_tokens",
                             "cost_usd", "seconds", "error"])

        for img in images:
            for run in range(1, args.repeats + 1):
                out_path = out_dir / f"{img.stem}_run{run}_fable.json"
                label = f"{img.name} (run {run})"

                if out_path.exists():
                    print(f"skip   {label}: already done")
                    continue
                if spent >= args.max_cost:
                    print(f"\nstopped: reached budget limit of ${args.max_cost:.2f}")
                    print(f"total estimated spend: ${spent:.4f}")
                    return

                start = time.time()
                try:
                    result, usage = extract(img, week_start)
                except SystemExit as e:          # bad file type / too large
                    writer.writerow([img.name, run, "skipped", "", "", "", "", "",
                                     "", "", str(e)])
                    print(f"skip   {label}: {e}")
                    break
                except Exception as e:           # API error, bad JSON, etc.
                    writer.writerow([img.name, run, "error", "", "", "", "", "",
                                     "", round(time.time() - start, 1), repr(e)])
                    f.flush()
                    print(f"ERROR  {label}: {e}")
                    continue
                seconds = time.time() - start

                out_path.write_text(json.dumps(result, indent=2))

                shifts = result["shifts"]
                ready = sum(1 for s in shifts if to_dayforce(s))
                low = sum(1 for s in shifts for c in s["confidence"].values() if c < 0.8)
                cost = (usage.input_tokens * PRICE_IN_PER_MTOK
                        + usage.output_tokens * PRICE_OUT_PER_MTOK) / 1e6
                spent += cost

                writer.writerow([img.name, run, "ok", len(shifts), ready, low,
                                 usage.input_tokens, usage.output_tokens,
                                 round(cost, 4), round(seconds, 1), ""])
                f.flush()
                print(f"done   {label}: {len(shifts)} shifts, ${cost:.4f}, "
                      f"{seconds:.0f}s  (total ${spent:.2f})")

    print(f"\nfinished. total estimated spend: ${spent:.4f}")
    print(f"results in: {out_dir}/   summary: {summary_path}")


if __name__ == "__main__":
    main()