from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable


DEFAULT_IGNORES = {
    ".git",
    ".ai-copilot",
    ".venv",
    "__pycache__",
    "node_modules",
}

DEFAULT_IGNORE_FILES = {
    "uv.lock",
    "poetry.lock",
    "package-lock.json",
    "pnpm-lock.yaml",
}

SOURCE_EXTENSIONS = {
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".java",
    ".go",
    ".rs",
    ".cs",
    ".cpp",
    ".c",
    ".h",
    ".hpp",
    ".md",
    ".toml",
}


def _is_allowed_path(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def read_file(path: str, root: Path) -> str:
    file_path = Path(path).expanduser().resolve()
    if not _is_allowed_path(file_path, root):
        raise ValueError("Path is outside the allowed workspace root")
    if not file_path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")
    return file_path.read_text(encoding="utf-8")


def search_code(query: str, root: Path, *, max_results: int = 20) -> list[str]:
    tokens = [
        token.strip(".,:;!?()[]{}\"'`/")
        for token in query.lower().split()
    ]
    tokens = [token for token in tokens if token]
    if not tokens:
        return []

    matches: list[str] = []
    fallback: list[str] = []
    for path in _iter_files(root):
        if path.name in DEFAULT_IGNORE_FILES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for idx, line in enumerate(text.splitlines(), start=1):
            haystack = line.lower()
            if any(token in haystack for token in tokens):
                entry = f"{path.relative_to(root)}:{idx}: {line.strip()}"
                if path.suffix.lower() in SOURCE_EXTENSIONS:
                    matches.append(entry)
                    if len(matches) >= max_results:
                        return matches
                else:
                    fallback.append(entry)
    matches.extend(fallback)
    return matches[:max_results]


def _iter_files(root: Path) -> Iterable[Path]:
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if name not in DEFAULT_IGNORES]
        for name in filenames:
            yield Path(dirpath) / name
