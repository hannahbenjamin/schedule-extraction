"""
Fable 5.1 state-of-the-art benchmark: extract one schedule image.

BENCHMARK ONLY. Uses an external API, so it is not part of the delivered POC
(see NFR-9). Use self-created synthetic schedules only - never Dayforce data.

Setup:
    pip install anthropic
    export ANTHROPIC_API_KEY="your-key"      # Windows: set ANTHROPIC_API_KEY=your-key

Run:
    python extract.py path/to/schedule.png
    python extract.py path/to/schedule.png 2026-11-09   # optional week start
"""

import base64
import json
import sys
from pathlib import Path
from datetime import date, timedelta

import anthropic

# configuration
MODEL = "claude-fable-5-1"
MAX_TOKENS = 16000

PRICE_IN_PER_MTOK = 10.00    # usd per million input tokens
PRICE_OUT_PER_MTOK = 50.00   # usd per million output tokens (includes thinking)

MAX_IMAGE_BYTES = 5 * 1024 * 1024

MEDIA_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
}

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday",
            "Friday", "Saturday", "Sunday"]

# output schema
FIELDS = [
    "employee_name",
    "clock_in_date",
    "clock_in_time",
    "clock_out_date",
    "clock_out_time",
    "job_assignment",
]

NULLABLE_STR = {"type": ["string", "null"]}

SCHEMA = {
    "type": "object",
    "properties": {
        "image_readable": {"type": "boolean"},  # fr-2a
        "shifts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    **{f: NULLABLE_STR for f in FIELDS},
                    "day_of_week": NULLABLE_STR,  # internal helper, cleaned up in code
                    "confidence": {
                        "type": "object",
                        "properties": {f: {"type": "number"} for f in FIELDS},
                        "required": FIELDS,
                        "additionalProperties": False,
                    },
                },
                "required": FIELDS + ["day_of_week", "confidence"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["image_readable", "shifts"],
    "additionalProperties": False,
}


# helpers
def build_prompt(week_start=None):
    year = date.today().year
    if week_start:
        date_hint = (f"- This schedule is for the week starting {week_start.isoformat()} "
                     f"({WEEKDAYS[week_start.weekday()]}). Use it to work out every date.")
    else:
        date_hint = ("- Work out each shift's date from anything in the image: column "
                     "headers, a title like 'Week of Nov 9', or a date range.\n"
                     f"- If the image shows month and day but no year, use {year}.\n"
                     "- If it only shows weekday names with no date anywhere, "
                     "leave the dates null.")

    return f"""Extract every shift from this workforce schedule image.

Rules:
- One entry per employee per shift.
- day_of_week: the full weekday name the shift starts on, one of Monday, Tuesday,
  Wednesday, Thursday, Friday, Saturday, Sunday. Fill this in whenever the schedule
  shows days, even if it shows no dates. Use null if the day is unknown.
- Dates as YYYY-MM-DD and times as HH:MM:SS in 24-hour format (ISO 8601).
{date_hint}
- Overnight shifts (e.g. 23:00 to 07:00): the clock-out date is the day after
  the clock-in date.
- If a field cannot be read or reasonably inferred, use null. Do not guess.
- confidence: a number from 0 to 1 for each field, how likely the value
  (including a null) is correct. Score each field separately; lower it for
  anything blurry, handwritten, crossed out or inferred.
- Set image_readable to false if the image is too poor to extract reliably."""


def load_image(path: Path):
    media_type = MEDIA_TYPES.get(path.suffix.lower())
    if media_type is None:
        sys.exit(f"Unsupported file type: {path.suffix}")
    data = path.read_bytes()
    if len(data) > MAX_IMAGE_BYTES:
        sys.exit(f"Image is {len(data)/1e6:.1f} MB; the limit is 5 MB. Resize it first.")
    return media_type, base64.b64encode(data).decode()


def normalize_day(value):
    """turn 'mon', 'Monday', 'MONDAY' etc. into 'Monday'; anything else -> None."""
    if not value:
        return None
    v = value.strip().lower()[:3]
    if len(v) < 2:
        return None
    for d in WEEKDAYS:
        if d.lower().startswith(v):
            return d
    return None


def fill_missing_dates(shift, week_start):
    """if the model gave a weekday but no date, compute the dates in code.
    returns True if any date was filled in by code (for benchmark transparency)."""
    filled = False
    day = normalize_day(shift["day_of_week"])
    shift["day_of_week"] = day
    if week_start is None or day is None:
        return filled
    if shift["clock_in_date"] is None:
        offset = (WEEKDAYS.index(day) - week_start.weekday()) % 7
        shift["clock_in_date"] = (week_start + timedelta(days=offset)).isoformat()
        filled = True
    if shift["clock_out_date"] is None and shift["clock_in_time"] and shift["clock_out_time"]:
        in_day = date.fromisoformat(shift["clock_in_date"])
        overnight = shift["clock_out_time"] <= shift["clock_in_time"]
        shift["clock_out_date"] = (in_day + timedelta(days=int(overnight))).isoformat()
        filled = True
    return filled


def extract(path: Path, week_start=None):
    media_type, img_b64 = load_image(path)
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY

    resp = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": media_type, "data": img_b64},
                    },
                    {"type": "text", "text": build_prompt(week_start)},
                ],
            }
        ],
    )

    # the response may include thinking blocks; the json is in the text block.
    text = next(b.text for b in resp.content if b.type == "text")
    result = json.loads(text)

    # keep the benchmark output shape stable while excluding mvp-only fields.
    for shift in result["shifts"]:
        shift["dates_filled_by_code"] = fill_missing_dates(shift, week_start)
        shift["meal_times"] = []  # fr-11: mvp exclusion
        shift["break_times"] = []  # fr-12: mvp exclusion
        shift["confidence"] = {
            k: min(max(v, 0.0), 1.0)
            for k, v in shift["confidence"].items()
        }

    return result, resp.usage


def to_dayforce(shift):
    """map one shift to the figure 1 payload. return none if required values are missing."""
    required = [
        "employee_name",
        "clock_in_date",
        "clock_in_time",
        "clock_out_date",
        "clock_out_time",
    ]
    if any(shift[f] is None for f in required):
        return None

    return {
        "TimeStart": f"{shift['clock_in_date']}T{shift['clock_in_time']}",
        "TimeEnd": f"{shift['clock_out_date']}T{shift['clock_out_time']}",
        "EmployeeXRefCode": shift["employee_name"],
        "PositionXRefCode": shift["job_assignment"],
    }


# main

def main():
    args = sys.argv[1:]
    if len(args) not in (1, 2):
        sys.exit("usage: python extract.py path/to/schedule.png [week-start YYYY-MM-DD]")

    path = Path(args[0])
    if not path.exists():
        sys.exit(f"file not found: {path}")

    week_start = None
    if len(args) == 2:
        try:
            week_start = date.fromisoformat(args[1])
        except ValueError:
            sys.exit("week start must be a date like 2026-11-09")

    print(f"extracting {path.name} with {MODEL}...")
    if week_start:
        print(f"week starting {week_start} ({WEEKDAYS[week_start.weekday()]})")
    result, usage = extract(path, week_start)

    # save the full benchmark output, including the internal confidence values.
    out_path = path.with_name(path.stem + "_fable.json")
    out_path.write_text(json.dumps(result, indent=2))

    # summarize the extraction results and estimated cost.
    shifts = result["shifts"]
    payload = [p for p in (to_dayforce(s) for s in shifts) if p]
    low_conf = [
        (i, f)
        for i, s in enumerate(shifts)
        for f, c in s["confidence"].items()
        if c < 0.8
    ]
    filled = sum(1 for s in shifts if s["dates_filled_by_code"])
    cost = (usage.input_tokens * PRICE_IN_PER_MTOK
            + usage.output_tokens * PRICE_OUT_PER_MTOK) / 1e6

    print(f"\nimage readable:      {result['image_readable']}")
    print(f"shifts found:        {len(shifts)}")
    print(f"ready for dayforce:  {len(payload)} (rest have missing required fields)")
    print(f"dates filled by code: {filled}")
    print(f"low-confidence:      {len(low_conf)} fields below 0.8")
    print(f"tokens:              {usage.input_tokens} in / {usage.output_tokens} out")
    print(f"estimated cost:      ${cost:.4f}")
    print(f"saved to:            {out_path}")

    print("\nfirst shifts:")
    for s in shifts[:5]:
        print(
            f"  {s['employee_name']!s:18} {s['day_of_week']!s:10}"
            f" {s['clock_in_date']} {s['clock_in_time']}"
            f" -> {s['clock_out_date']} {s['clock_out_time']}  {s['job_assignment']}"
        )


if __name__ == "__main__":
    main()