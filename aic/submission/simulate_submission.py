"""Generate fake final rows and verify the BTC submission ZIP structure.

Run from the repository root:
    python -m aic.submission.simulate_submission
"""

import csv
import io
import zipfile
from pathlib import Path

if __package__:
    from .writer import write_query_submission
else:
    # Support: cd aic/submission && python simulate_submission.py
    from writer import write_query_submission


QUERY_ID = "pack1_q3_kis"
OUTPUT_PATH = Path(__file__).resolve().parent / "submission.zip"


def generate_rows(count: int = 100) -> list[list[str]]:
    """Simulate the final 1-based rows received by the submission module."""
    return [
        [f"L{21 + index // 10:02d}_V{index + 1:03d}", str(100 + index * 30)]
        for index in range(count)
    ]


def assert_valid_submission(zip_path: Path, query_id: str, expected_rows) -> None:
    expected_csv_path = f"submission/{query_id}.csv"

    assert zip_path.is_file(), f"ZIP was not created: {zip_path}"
    with zipfile.ZipFile(zip_path) as archive:
        assert archive.namelist() == [expected_csv_path], (
            f"Wrong ZIP structure: {archive.namelist()}"
        )
        raw = archive.read(expected_csv_path)

    assert not raw.startswith(b"\xef\xbb\xbf"), "CSV must be UTF-8 without BOM"
    actual_rows = list(csv.reader(io.StringIO(raw.decode("utf-8"))))
    assert actual_rows == expected_rows, "CSV rows differ from the input list"
    assert 0 < len(actual_rows) <= 100, "BTC allows 1 to 100 rows per query"


def main() -> None:
    rows = generate_rows()
    write_query_submission(QUERY_ID, rows, OUTPUT_PATH)
    assert_valid_submission(OUTPUT_PATH, QUERY_ID, rows)

    print(f"PASS: created {OUTPUT_PATH}")
    print(f"PASS: contains submission/{QUERY_ID}.csv")
    print(f"PASS: {len(rows)} rows, UTF-8 without BOM")


if __name__ == "__main__":
    main()
