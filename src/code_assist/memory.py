from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


MEMORY_FILENAME = "memory.jsonl"


@dataclass(frozen=True)
class MemoryEntry:
    timestamp: str
    question: str
    answer: str


def memory_path(root: Path) -> Path:
    return root / ".ai-copilot" / MEMORY_FILENAME


def append_memory(root: Path, question: str, answer: str) -> None:
    path = memory_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = MemoryEntry(
        timestamp=datetime.now(timezone.utc).isoformat(),
        question=question,
        answer=answer,
    )
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry.__dict__, ensure_ascii=True) + "\n")


def load_memory(root: Path, *, limit: int = 10) -> list[MemoryEntry]:
    path = memory_path(root)
    if not path.exists():
        return []

    entries: list[MemoryEntry] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        entries.append(
            MemoryEntry(
                timestamp=str(payload.get("timestamp", "")),
                question=str(payload.get("question", "")),
                answer=str(payload.get("answer", "")),
            )
        )
    return entries[-limit:]


def format_memory(entries: Iterable[MemoryEntry]) -> str:
    lines: list[str] = []
    for entry in entries:
        lines.append(f"Q: {entry.question}")
        lines.append(f"A: {entry.answer}")
    return "\n".join(lines)
