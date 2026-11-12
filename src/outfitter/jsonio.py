from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def canonical_json(value: Any) -> str:
    """Serialize a value deterministically for hashes and equality checks."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def read_json_object(path: str | Path) -> dict[str, Any]:
    """Read a JSON document whose root must be an object."""

    target = Path(path)
    value = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {target}")
    return value


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read non-empty JSONL records and require object-shaped rows."""

    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"invalid JSONL at line {line_number}: {error}"
            ) from error
        if not isinstance(value, dict):
            raise ValueError(f"JSONL line {line_number} is not an object")
        rows.append(value)
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write deterministic UTF-8 JSONL with exactly one trailing newline."""

    payload = "".join(
        json.dumps(
            row,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
        for row in rows
    )
    Path(path).write_text(payload, encoding="utf-8")


def write_json(path: str | Path, value: Any) -> None:
    """Write the project's deterministic, human-readable JSON format."""

    Path(path).write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def write_jsonl_atomic(
    path: str | Path, rows: Iterable[dict[str, Any]]
) -> None:
    target = Path(path)
    temporary = target.with_suffix(target.suffix + ".tmp")
    write_jsonl(temporary, rows)
    temporary.replace(target)


def write_json_atomic(path: str | Path, value: Any) -> None:
    target = Path(path)
    temporary = target.with_suffix(target.suffix + ".tmp")
    write_json(temporary, value)
    temporary.replace(target)
