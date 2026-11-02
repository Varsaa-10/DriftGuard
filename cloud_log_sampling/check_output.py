import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Validate that a merged CloudTrail JSON file loads correctly.")
    parser.add_argument("json_file", type=Path, help="Path to the JSON file to validate")
    args = parser.parse_args()

    if not args.json_file.exists():
        print(f"File not found: {args.json_file}")
        sys.exit(1)

    try:
        with open(args.json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Invalid JSON: {e}")
        sys.exit(1)

    if not isinstance(data, dict) or "Records" not in data:
        print("Invalid structure: expected a top-level object with a 'Records' key.")
        sys.exit(1)

    records = data["Records"]
    if not isinstance(records, list):
        print("Invalid structure: 'Records' should be a list.")
        sys.exit(1)

    non_dict_records = [i for i, r in enumerate(records) if not isinstance(r, dict)]
    if non_dict_records:
        print(f"Invalid structure: {len(non_dict_records)} record(s) are not objects, e.g. index {non_dict_records[0]}.")
        sys.exit(1)

    missing_event_name = sum(1 for r in records if not r.get("eventName"))

    print(f"OK: {args.json_file} is valid JSON.")
    print(f"Records: {len(records)}")
    if missing_event_name:
        print(f"Warning: {missing_event_name} record(s) missing 'eventName'.")


if __name__ == "__main__":
    main()
