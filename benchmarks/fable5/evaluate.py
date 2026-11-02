"""
Score Fable benchmark outputs against ground truth, using the PR Evaluation Plan
(Table 6 / Appendix A, Table 9).

Put this file in benchmarks/fable5/ next to run_benchmark.py.

Run (after run_benchmark.py):
    python evaluate.py
    python evaluate.py --data data --results results --threshold 0.8

How files are matched:
    For every image in data/ (subfolders included), the ground truth is the
    *_ground_truth.json (or *_truth.json) file in the SAME folder. Its results
    are results/<image name>_run<N>_fable.json.

Outputs (in the results folder):
    metrics_fields.csv   precision / recall / F1 / FP / FN per field   (Field Extraction)
    metrics_images.csv   one row per image run                         (all areas)
    printed report       PR metrics vs. PR benchmark targets

Metric definitions (state these in your report):
    Shifts are first paired between output and truth (a pair needs >= 3 of the
    6 fields to agree). Unpaired output shifts are extra (hallucinated) shifts;
    unpaired truth shifts are missed shifts.
    For each field:
      TP  output has a value and it is correct
      FP  output has a value and it is wrong (or the field/shift does not exist)
      FN  truth has a value and output is null or wrong
      TN  both are null
    A wrong value counts as both FP and FN (standard for field extraction).
    precision = TP/(TP+FP)   recall = TP/(TP+FN)   F1 = harmonic mean
    accuracy  = (TP+TN) / all field slots
    FP rate   = FP/(TP+FP)  share of extracted values that are wrong
    FN rate   = FN/(TP+FN)  share of true values that were missed
"""

import argparse
import csv
import json
import re
from collections import defaultdict
from datetime import date
from pathlib import Path

FIELDS = ["employee_name", "clock_in_date", "clock_in_time",
          "clock_out_date", "clock_out_time", "job_assignment"]
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
MATCH_MIN_FIELDS = 3

# PR Appendix A benchmark references
TARGETS = {
    "flagged_incorrect": 0.90,    # >= 90% of incorrect fields flagged
    "empty_as_null": 0.85,        # >= 85% of empty fields returned as null
    "hallucinated": 0.10,         # <= 10% hallucinated fields
    "printed_accuracy": 0.90,     # >= 90% on printed text (reference only)
    "handwritten_accuracy": 0.70, # ~70% on handwriting (reference only)
}


# ------------------------------------------------------------ normalising
def norm(field, value):
    """Make values comparable: ignore case/extra spaces, pad times, check dates."""
    if value is None:
        return None
    v = str(value).strip()
    if v == "" or v.lower() in {"null", "none", "n/a"}:
        return None
    if field.endswith("_time"):
        m = re.fullmatch(r"(\d{1,2}):(\d{2})(?::(\d{2}))?", v)
        if m:
            return f"{int(m[1]):02d}:{m[2]}:{m[3] or '00'}"
        return v
    if field.endswith("_date"):
        try:
            return date.fromisoformat(v).isoformat()
        except ValueError:
            return v
    return re.sub(r"\s+", " ", v).lower()


def load_shifts(path):
    data = json.loads(Path(path).read_text())
    shifts = data["shifts"] if isinstance(data, dict) else data
    missing = [f for f in FIELDS if shifts and f not in shifts[0]]
    if missing:
        raise ValueError(f"{path} is missing fields {missing}. "
                         f"Expected keys: {FIELDS}")
    return data, shifts


# ------------------------------------------------------------ matching
def pair_shifts(pred, truth):
    """Greedy pairing by number of agreeing fields. Returns pairs and leftovers."""
    scores = []
    for i, p in enumerate(pred):
        for j, t in enumerate(truth):
            s = sum(norm(f, p[f]) == norm(f, t[f]) and norm(f, t[f]) is not None
                    for f in FIELDS)
            if s >= MATCH_MIN_FIELDS:
                scores.append((s, i, j))
    scores.sort(reverse=True)
    used_p, used_t, pairs = set(), set(), []
    for s, i, j in scores:
        if i not in used_p and j not in used_t:
            used_p.add(i); used_t.add(j); pairs.append((i, j))
    extra = [i for i in range(len(pred)) if i not in used_p]
    missed = [j for j in range(len(truth)) if j not in used_t]
    return pairs, extra, missed


# ------------------------------------------------------------ scoring
def new_counts():
    return {"TP": 0, "FP": 0, "FN": 0, "TN": 0, "slots": 0,
            "wrong": 0, "wrong_flagged": 0, "right": 0, "right_flagged": 0,
            "truth_null": 0, "truth_null_kept_null": 0,
            "pred_values": 0, "hallucinated": 0}


def score_field(c, field, p_shift, t_shift, threshold):
    pv = norm(field, p_shift[field]) if p_shift else None
    tv = norm(field, t_shift[field]) if t_shift else None
    conf = (p_shift or {}).get("confidence", {}).get(field)
    flagged = conf is not None and conf < threshold

    c["slots"] += 1
    if pv is not None:
        c["pred_values"] += 1
    if tv is None:
        c["truth_null"] += t_shift is not None
        if pv is None:
            c["TN"] += 1
            c["truth_null_kept_null"] += t_shift is not None
        else:
            c["FP"] += 1
            c["hallucinated"] += 1           # value where none exists
    else:
        if pv == tv:
            c["TP"] += 1
        else:
            c["FN"] += 1
            if pv is not None:
                c["FP"] += 1

    if p_shift is not None:                   # confidence checks need an output field
        correct = pv == tv
        if correct:
            c["right"] += 1
            c["right_flagged"] += flagged
        else:
            c["wrong"] += 1
            c["wrong_flagged"] += flagged


def score_run(pred, truth, threshold):
    """Returns per-field counts and shift-level numbers for one run."""
    pairs, extra, missed = pair_shifts(pred, truth)
    counts = {f: new_counts() for f in FIELDS}
    for i, j in pairs:
        for f in FIELDS:
            score_field(counts[f], f, pred[i], truth[j], threshold)
    for i in extra:
        for f in FIELDS:
            score_field(counts[f], f, pred[i], None, threshold)
    for j in missed:
        for f in FIELDS:
            score_field(counts[f], f, None, truth[j], threshold)
    exact = sum(all(norm(f, pred[i][f]) == norm(f, truth[j][f]) for f in FIELDS)
                for i, j in pairs)
    shift = {"pred": len(pred), "truth": len(truth), "matched": len(pairs),
             "extra": len(extra), "missed": len(missed), "exact": exact}
    return counts, shift


def add(total, part):
    for k, v in part.items():
        total[k] += v


def ratio(a, b):
    return a / b if b else None


def prf(c):
    p = ratio(c["TP"], c["TP"] + c["FP"])
    r = ratio(c["TP"], c["TP"] + c["FN"])
    if p is None or r is None:
        f1 = None
    else:
        f1 = 2 * p * r / (p + r) if p + r else 0.0
    acc = ratio(c["TP"] + c["TN"], c["slots"])
    return p, r, f1, acc


def comparable(shifts):
    return sorted(json.dumps({f: norm(f, s[f]) for f in FIELDS}, sort_keys=True)
                  for s in shifts)


def pct(x):
    return "  n/a " if x is None else f"{100 * x:5.1f}%"


# ------------------------------------------------------------ main
def find_truth(img):
    cands = sorted(set(img.parent.glob("*ground_truth.json")) |
                   set(img.parent.glob("*_truth.json")))
    own = img.parent / f"{img.stem}_truth.json"
    if own.exists():
        return own
    return cands[0] if len(cands) == 1 else None


def read_summary(results):
    path = results / "summary.csv"
    info = {}
    if path.exists():
        with open(path) as f:
            for row in csv.DictReader(f):
                info[(Path(row["image"]).stem, row["run"])] = row
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--results", default="results")
    ap.add_argument("--threshold", type=float, default=0.8,
                    help="confidence below this counts as 'flagged'")
    args = ap.parse_args()

    data_dir, res_dir = Path(args.data), Path(args.results)
    summary = read_summary(res_dir)
    images = sorted(p for p in data_dir.rglob("*") if p.suffix.lower() in IMAGE_EXT)

    field_totals = {f: new_counts() for f in FIELDS}
    shift_totals = defaultdict(int)
    image_rows, warnings = [], []
    readable = runs_scored = 0
    consistency = []          # (image, identical_to_run1)
    seconds_by_size = []

    for img in images:
        truth_path = find_truth(img)
        run_files = sorted(res_dir.glob(f"{img.stem}_run*_fable.json"),
                           key=lambda p: int(re.search(r"_run(\d+)_", p.name)[1]))
        if not run_files:
            warnings.append(f"no results for {img.name}")
            continue
        if truth_path is None:
            warnings.append(f"no ground truth found next to {img.name}")
            continue
        _, truth = load_shifts(truth_path)

        first = None
        for rf in run_files:
            run = re.search(r"_run(\d+)_", rf.name)[1]
            out, pred = load_shifts(rf)
            counts, shift = score_run(pred, truth, args.threshold)
            for f in FIELDS:
                add(field_totals[f], counts[f])
            add(shift_totals, shift)
            runs_scored += 1
            readable += bool(out.get("image_readable", True)) if isinstance(out, dict) else 1

            run_total = new_counts()
            for f in FIELDS:
                add(run_total, counts[f])
            p, r, f1, acc = prf(run_total)

            comp = comparable(pred)
            if first is None:
                first = comp
            else:
                consistency.append((img.name, comp == first))

            s = summary.get((img.stem, run), {})
            secs = float(s["seconds"]) if s.get("seconds") else None
            if secs is not None:
                seconds_by_size.append((len(truth), secs))

            image_rows.append({
                "image": img.name, "run": run,
                "truth_shifts": shift["truth"], "output_shifts": shift["pred"],
                "missed_shifts": shift["missed"], "extra_shifts": shift["extra"],
                "exact_shifts": shift["exact"],
                "field_precision": p, "field_recall": r, "field_f1": f1,
                "field_accuracy": acc,
                "seconds": secs, "cost_usd": s.get("cost_usd"),
            })

    if not runs_scored:
        print("nothing to score.")
        for w in warnings:
            print("  " + w)
        return

    # ---------------- write CSVs
    with open(res_dir / "metrics_fields.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["field", "TP", "FP", "FN", "TN", "precision", "recall", "f1",
                    "accuracy", "fp_rate", "fn_rate"])
        for fld in FIELDS:
            c = field_totals[fld]
            p, r, f1, acc = prf(c)
            w.writerow([fld, c["TP"], c["FP"], c["FN"], c["TN"], p, r, f1, acc,
                        ratio(c["FP"], c["TP"] + c["FP"]),
                        ratio(c["FN"], c["TP"] + c["FN"])])
    with open(res_dir / "metrics_images.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(image_rows[0]))
        w.writeheader()
        w.writerows(image_rows)

    # ---------------- report
    all_c = new_counts()
    for fld in FIELDS:
        add(all_c, field_totals[fld])
    P, R, F1, ACC = prf(all_c)
    st = shift_totals

    errors = sum(1 for row in summary.values() if row.get("status") == "error")
    attempts = len(summary) if summary else runs_scored

    print(f"\nscored {runs_scored} runs across {len(images)} images "
          f"(confidence threshold {args.threshold})\n")

    print("FIELD EXTRACTION (per field)")
    print(f"  {'field':16} {'precision':>9} {'recall':>8} {'F1':>8} {'accuracy':>9}"
          f" {'FP rate':>8} {'FN rate':>8}")
    for fld in FIELDS:
        c = field_totals[fld]
        p, r, f1, acc = prf(c)
        print(f"  {fld:16} {pct(p):>9} {pct(r):>8} {pct(f1):>8} {pct(acc):>9}"
              f" {pct(ratio(c['FP'], c['TP'] + c['FP'])):>8}"
              f" {pct(ratio(c['FN'], c['TP'] + c['FN'])):>8}")
    print(f"  {'ALL FIELDS':16} {pct(P):>9} {pct(R):>8} {pct(F1):>8} {pct(ACC):>9}")
    print(f"  reference: >=90% printed text, ~70% handwriting (Appendix A)\n")

    print("SHIFTS")
    print(f"  shift recall (found / in truth):        {pct(ratio(st['matched'], st['truth']))}"
          f"   ({st['matched']}/{st['truth']})")
    print(f"  shift precision (real / output):        {pct(ratio(st['matched'], st['pred']))}"
          f"   ({st['extra']} extra shifts)")
    print(f"  shifts with every field correct:        {pct(ratio(st['exact'], st['truth']))}\n")

    print("CONFIDENCE (FR-14)                     result   target")
    print(f"  incorrect fields flagged (<{args.threshold}):     "
          f"{pct(ratio(all_c['wrong_flagged'], all_c['wrong']))}   >=90%"
          f"   ({all_c['wrong']} incorrect fields)")
    print(f"  empty fields returned as null:        "
          f"{pct(ratio(all_c['truth_null_kept_null'], all_c['truth_null']))}   >=85%"
          f"   ({all_c['truth_null']} empty fields in truth)")
    print(f"  hallucinated fields:                  "
          f"{pct(ratio(all_c['hallucinated'], all_c['pred_values']))}   <=10%")
    print(f"  correct fields wrongly flagged:       "
          f"{pct(ratio(all_c['right_flagged'], all_c['right']))}   (lower is better)\n")

    print("INPUT / OUTPUT / RELIABILITY")
    print(f"  marked readable (FR-2a):              {pct(ratio(readable, runs_scored))}")
    print(f"  runs with valid JSON:                 {pct(ratio(runs_scored, attempts))}"
          f"   target 100%")
    print(f"  crash/error rate:                     {pct(ratio(errors, attempts))}")
    if consistency:
        same = sum(ok for _, ok in consistency)
        print(f"  repeat runs identical to run 1 (NFR-8): "
              f"{pct(ratio(same, len(consistency)))}   ({same}/{len(consistency)})")
    else:
        print("  repeat runs (NFR-8):                  n/a  (use --repeats 3 in run_benchmark.py)")

    if seconds_by_size:
        print("\nPERFORMANCE (time by schedule size)")
        by = defaultdict(list)
        for n, s in seconds_by_size:
            by[n].append(s)
        for n in sorted(by):
            print(f"  {n:4d} shifts: {sum(by[n]) / len(by[n]):6.1f} s average"
                  f" over {len(by[n])} run(s)")

    if warnings:
        print("\nwarnings:")
        for w in warnings:
            print("  " + w)
    print(f"\nsaved: {res_dir / 'metrics_fields.csv'}, {res_dir / 'metrics_images.csv'}")
    print("not covered here: structure recognition (Fable has no separate grid step),"
          "\nAPI acceptance, usability, security and maintainability reviews.")


if __name__ == "__main__":
    main()