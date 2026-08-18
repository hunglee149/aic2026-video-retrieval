import argparse
import json
from pathlib import Path

from .writer import write_query_submission


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a JSON list of rows into a BTC submission ZIP."
    )
    parser.add_argument(
        "--query-id",
        required=True,
        help="BTC query filename without its extension, for example pack1_q3_kis",
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help='JSON file containing rows such as [["L21_V001", "712"]]',
    )
    parser.add_argument(
        "--output",
        default=Path("submission.zip"),
        type=Path,
        help="Output ZIP path (default: submission.zip)",
    )
    args = parser.parse_args()

    with args.input.open(encoding="utf-8") as source:
        rows = json.load(source)
    if not isinstance(rows, list) or any(not isinstance(row, list) for row in rows):
        parser.error("input JSON must be a list of row lists")

    result = write_query_submission(args.query_id, rows, args.output)
    print(result)


if __name__ == "__main__":
    main()
