"""Download the raw historical data (2010-11 through 2023-24 regular seasons).

Source: https://github.com/NocturneBear/NBA-Data-2010-2024 (MIT licensed),
which mirrors NBA.com box scores. ~80 MB total.

Usage:  python src/get_data.py
"""
from pathlib import Path
import urllib.request

BASE = "https://raw.githubusercontent.com/NocturneBear/NBA-Data-2010-2024/main/"
FILES = [
    "regular_season_box_scores_2010_2024_part_1.csv",
    "regular_season_box_scores_2010_2024_part_2.csv",
    "regular_season_box_scores_2010_2024_part_3.csv",
    "regular_season_totals_2010_2024.csv",
]
RAW = Path(__file__).resolve().parents[1] / "data" / "raw"


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        dest = RAW / name
        if dest.exists():
            print(f"skip  {name} (already downloaded)")
            continue
        print(f"get   {name} ...", flush=True)
        urllib.request.urlretrieve(BASE + name, dest)
    print("done ->", RAW)


if __name__ == "__main__":
    main()
