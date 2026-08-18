import csv
import io
import zipfile

# NHO FIX LAI NGHEN CAI NAY TEST THOI

def write_submission(rows_by_query, out_path):
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for query_id, rows in rows_by_query.items():
            buffer = io.StringIO(newline="")
            csv.writer(buffer).writerows(rows)
            archive.writestr(
                "submission/%s.csv" % query_id,
                buffer.getvalue().encode("utf-8"),
            )
    return str(out_path)
