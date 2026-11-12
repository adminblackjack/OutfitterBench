from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    target = Path(path)
    fields = list(rows[0]) if rows else []
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
