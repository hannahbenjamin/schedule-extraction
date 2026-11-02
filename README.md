## Repository structure

```
.
├── benchmarks/
│   └── fable5/              State-of-the-art reference (external API, synthetic data only)
│       ├── data/            Synthetic test schedules: S01/ … S10/
│       │   └── S01/
│       │       ├── S01_printed.png
│       │       └── S01_ground_truth.json
│       ├── extract.py       Extracts one image with Claude Fable 5.1
│       ├── run_benchmark.py Runs extract.py over a folder of images
│       ├── evaluate.py      Scores outputs against ground truth (PR evaluation metrics)
│       ├── test_api.py      Checks the API key and model access
│       └── requirements.txt
├── pipelines/               Candidate POC pipelines (run inside the client environment)
├── .gitignore
└── README.md
```

Each pipeline under `pipelines/` and each benchmark under `benchmarks/` produces output in
the same schema, so all approaches are scored with the same evaluation code.

---

## Setup

Requires Python 3.10+.

```bash
git clone <repo-url>
cd <repo-folder>
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
```

### Fable benchmark

```bash
cd benchmarks/fable5
pip install -r requirements.txt
export ANTHROPIC_API_KEY="your-key"  # Windows: set ANTHROPIC_API_KEY=your-key
python test_api.py                   # optional: confirm the key works
```

Never commit the key. Use a `.env` file (gitignored) or set it in the terminal.

---

## Running the benchmark

All commands run from `benchmarks/fable5/`.

**One image**

```bash
python extract.py data/S01/S01_printed.png
python extract.py data/S03/S03_weekdays.png 2026-11-09   # schedule shows weekdays only
```

**A folder of images** (subfolders included)

```bash
python run_benchmark.py data
python run_benchmark.py data --repeats 3 --max-cost 5
```

| Option | Meaning | Default |
|---|---|---|
| `--week-start YYYY-MM-DD` | Week start for schedules that show weekdays but no dates | none |
| `--repeats N` | Runs per image, for the NFR-8 consistency check | 1 |
| `--max-cost USD` | Stop once estimated spend reaches this amount | 5 |
| `--out FOLDER` | Where results are written | `results` |

Finished runs are skipped, so an interrupted run can be restarted safely.

**Selected schedules only** (e.g. S01–S05)

```bash
for d in data/S0{1..5}; do python run_benchmark.py "$d" --max-cost 1; done
```

**Scoring**

```bash
python evaluate.py | tee results/metrics_report.txt
```

---

## Outputs

Written to `benchmarks/fable5/results/` (gitignored):

| File | Contents |
|---|---|
| `<image>_run<N>_fable.json` | Extracted shifts for one run |
| `summary.csv` | Per run: shifts found, tokens, cost, time, errors |
| `metrics_fields.csv` | Per field: TP, FP, FN, TN, precision, recall, F1, accuracy, FP/FN rate |
| `metrics_images.csv` | Per image and run: shift and field metrics, time, cost |
| `metrics_report.txt` | Full printed report (when saved with `tee`) |

To keep final results for a report, copy them outside `results/` before committing.

---

## Evaluation

`evaluate.py` implements the field-level parts of the PR Evaluation Plan (Table 6 and
Appendix A):

| Area | Metrics | PR reference |
|---|---|---|
| Field extraction | Precision, recall, F1, accuracy, FP/FN rate per field | ≥90% printed, ~70% handwriting |
| Confidence (FR-14) | Incorrect fields flagged; empty fields returned as null; hallucinated fields | ≥90%, ≥85%, ≤10% |
| Output | Valid JSON rate | 100% |
| Reliability | Error rate; repeat runs identical (NFR-8) | Consistent output |
| Performance | Time by schedule size | ~4.2 s/page |

Metric definitions (TP, FP, FN, matching rules) are documented at the top of `evaluate.py`.
Structure recognition, API acceptance, usability, security and maintainability are
evaluated separately.

---

## Test data

- Every schedule folder contains the image(s) and a `_ground_truth.json` with the correct
  answers in the output schema above.
- All names and shifts are invented. Do not add real schedules, real names or client data.
- Keep images small; for large data sets, commit the generator script instead of the images.