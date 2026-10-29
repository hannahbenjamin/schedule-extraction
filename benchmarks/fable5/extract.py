"""
Fable 5.1 state-of-the-art benchmark: extract one schedule image.

BENCHMARK ONLY. Uses an external API, so it is not part of the delivered POC
(see NFR-9). Use self-created synthetic schedules only - never Dayforce data.

Setup:
    pip install anthropic
    export ANTHROPIC_API_KEY="your-key"      # Windows: set ANTHROPIC_API_KEY=your-key

Run:
    python fable_extract.py path/to/schedule.png
"""

import base64
import json
import sys
from pathlib import Path

import anthropic

# ---------------------------------------------------------------- settings
MODEL = "claude-fable-5-1"
MAX_TOKENS = 16000

PRICE_IN_PER_MTOK = 10.00    
PRICE_OUT_PER_MTOK = 50.00   

MAX_IMAGE_BYTES = 5 * 1024 * 1024 

MEDIA_TYPES = {".png": "image/png", ".jpg": "image/jpeg",
               ".jpeg": "image/jpeg", ".webp": "image/webp", ".gif": "image/gif"}

# -------------------------------------------- output schema 
FIELDS = ["employee_name", "clock_in_date", "clock_in_time",
          "clock_out_date", "clock_out_time", "job_assignment"]

NULLABLE_STR = {"type": ["string", "null"]}

SCHEMA = {
    "type": "object",
    "properties": {
        "image_readable": {"type": "boolean"},          # FR-2a
        "shifts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    **{f: NULLABLE_STR for f in FIELDS},
                    "confidence": {
                        "type": "object",
                        "properties": {f: {"type": "number"} for f in FIELDS},
                        "required": FIELDS,
                        "additionalProperties": False,
                    },
                },
                "required": FIELDS + ["confidence"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["image_readable", "shifts"],
    "additionalProperties": False,
}

PROMPT = """Extract every shift from this workforce schedule image.

Rules:
- One entry per employee per shift.
- Dates as YYYY-MM-DD and times as HH:MM:SS in 24-hour format (ISO 8601).
- If a field cannot be read or reasonably inferred, use null. Do not guess.
- Logical inference is allowed. Example: for an overnight shift (23:00 to 07:00),
  the clock-out date is the day after the clock-in date.
- If the schedule shows weekdays but no year or dates, leave dates null.
- confidence: a number from 0 to 1 for each field, how likely the value
  (including a null) is correct.
- Set image_readable to false if the image is too poor to extract reliably."""


# ---------------------------------------------------------------- helpers
def load_image(path: Path):
    media_type = MEDIA_TYPES.get(path.suffix.lower())
    if media_type is None:
        sys.exit(f"Unsupported file type: {path.suffix}")
    data = path.read_bytes()
    if len(data) > MAX_IMAGE_BYTES:
        sys.exit(f"Image is {len(data)/1e6:.1f} MB; the limit is 5 MB. Resize it first.")
    return media_type, base64.b64encode(data).decode()


def extract(path: Path):
    media_type, img_b64 = load_image(path)
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY

    resp = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
        messages=[{
            "role": "user",
            "content": [
                {"type": "image",
                 "source": {"type": "base64", "media_type": media_type, "data": img_b64}},
                {"type": "text", "text": PROMPT},
            ],
        }],
    )

    # The response may include thinking blocks; the JSON is in the text block.
    text = next(b.text for b in resp.content if b.type == "text")
    result = json.loads(text)

    # Post-processing per the PR
    for shift in result["shifts"]:
        shift["meal_times"] = []    # FR-11: MVP exclusion
        shift["break_times"] = []   # FR-12: MVP exclusion
        shift["confidence"] = {k: min(max(v, 0.0), 1.0)
                               for k, v in shift["confidence"].items()}

    return result, resp.usage


def to_dayforce(shift):
    """Map one shift to the Figure 1 payload. None if required values are missing."""
    required = ["employee_name", "clock_in_date", "clock_in_time",
                "clock_out_date", "clock_out_time"]
    if any(shift[f] is None for f in required):
        return None
    return {
        "TimeStart": f"{shift['clock_in_date']}T{shift['clock_in_time']}",
        "TimeEnd": f"{shift['clock_out_date']}T{shift['clock_out_time']}",
        "EmployeeXRefCode": shift["employee_name"],
        "PositionXRefCode": shift["job_assignment"],
    }


# ------------------------------------------------------------------- main
def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python fable_extract.py path/to/schedule.png")
    path = Path(sys.argv[1])
    if not path.exists():
        sys.exit(f"File not found: {path}")

    print(f"Extracting {path.name} with {MODEL}...")
    result, usage = extract(path)

    # Save full result (Table 3 format, including internal confidence)
    out_path = path.with_name(path.stem + "_fable.json")
    out_path.write_text(json.dumps(result, indent=2))

    # Summary
    shifts = result["shifts"]
    payload = [p for p in (to_dayforce(s) for s in shifts) if p]
    low_conf = [(i, f) for i, s in enumerate(shifts)
                for f, c in s["confidence"].items() if c < 0.8]
    cost = (usage.input_tokens * PRICE_IN_PER_MTOK
            + usage.output_tokens * PRICE_OUT_PER_MTOK) / 1e6

    print(f"\nImage readable:      {result['image_readable']}")
    print(f"Shifts found:        {len(shifts)}")
    print(f"Ready for Dayforce:  {len(payload)} (rest have missing required fields)")
    print(f"Low-confidence:      {len(low_conf)} fields below 0.8")
    print(f"Tokens:              {usage.input_tokens} in / {usage.output_tokens} out")
    print(f"Estimated cost:      ${cost:.4f}")
    print(f"Saved to:            {out_path}")

    print("\nFirst shifts:")
    for s in shifts[:5]:
        print(f"  {s['employee_name']!s:20} {s['clock_in_date']} {s['clock_in_time']}"
              f" -> {s['clock_out_date']} {s['clock_out_time']}  {s['job_assignment']}")


if __name__ == "__main__":
    main()