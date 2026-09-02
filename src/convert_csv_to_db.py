"""
Convert csv file to an sql database
"""

import csv
import sqlite3
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = _ROOT / "data" / "battery_features_clean.csv"
OUTPUT_DB = _ROOT / "data" / "battery_degradation_clean.db"
TABLE_NAME = "battery_degradation_clean"


def infer_sql_type(values):
    non_empty = [v for v in values if v not in (None, "")]
    if not non_empty:
        return "TEXT"

    all_int = True
    all_float = True

    for value in non_empty:
        try:
            int(value)
        except ValueError:
            all_int = False

        try:
            float(value)
        except ValueError:
            all_float = False

    if all_int:
        return "INTEGER"
    if all_float:
        return "REAL"
    return "TEXT"


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"CSV file not found: {INPUT_CSV}")

    with INPUT_CSV.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if not reader.fieldnames:
            raise ValueError(f"CSV file is empty or has no header row: {INPUT_CSV}")

        rows = list(reader)

    column_types = {}
    for column in reader.fieldnames:
        values = [row.get(column, "") for row in rows]
        column_types[column] = infer_sql_type(values)

    columns_sql = ", ".join(
        f'"{column}" {column_types[column]}' for column in reader.fieldnames
    )

    connection = sqlite3.connect(OUTPUT_DB)
    try:
        cursor = connection.cursor()
        cursor.execute(f"DROP TABLE IF EXISTS \"{TABLE_NAME}\";")
        cursor.execute(f"CREATE TABLE \"{TABLE_NAME}\" ({columns_sql});")

        insert_columns = ", ".join(f'"{column}"' for column in reader.fieldnames)
        placeholders = ", ".join("?" for _ in reader.fieldnames)
        insert_sql = f"INSERT INTO \"{TABLE_NAME}\" ({insert_columns}) VALUES ({placeholders})"

        data = [
            tuple(row.get(column, "") for column in reader.fieldnames)
            for row in rows
        ]

        cursor.executemany(insert_sql, data)
        connection.commit()

        print(f"Converted {INPUT_CSV.name} to {OUTPUT_DB.name}")
        print(f"Rows inserted: {len(data)}")
        print(f"Table created: {TABLE_NAME}")
    finally:
        connection.close()


if __name__ == "__main__":
    main()
