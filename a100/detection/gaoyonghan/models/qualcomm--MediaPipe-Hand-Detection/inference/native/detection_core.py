"""Local data and output helpers for this model; copied from the verified runner."""
from __future__ import annotations
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"损坏的断点文件 {path}:{line_number}") from exc
    ids = [record["image_id"] for record in records]
    if len(ids) != len(set(ids)):
        raise RuntimeError(f"断点文件包含重复 image_id: {path}")
    return records


def ensure_coco_fields(annotation: dict[str, Any]) -> dict[str, Any]:
    annotation.setdefault("info", {})
    annotation.setdefault("licenses", [])
    annotation.setdefault("annotations", [])
    return annotation
