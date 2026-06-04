from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

DEFAULT_EXTENSIONS = {
    ".c",
    ".cc",
    ".cpp",
    ".cs",
    ".go",
    ".h",
    ".hpp",
    ".java",
    ".js",
    ".json",
    ".jsx",
    ".md",
    ".py",
    ".rs",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".yaml",
    ".yml",
}

DEFAULT_IGNORES = {
    ".git",
    ".ai-copilot",
    ".venv",
    "__pycache__",
    "node_modules",
    "dist",
    "build",
    ".mypy_cache",
    ".pytest_cache",
}


def ingest_repo(
    root: Path,
    persist_dir: Path,
    embeddings_model: str,
    *,
    chunk_size: int = 800,
    chunk_overlap: int = 100,
    max_file_size_kb: int = 512,
) -> int:
    files = list(_iter_files(root, DEFAULT_EXTENSIONS, DEFAULT_IGNORES))
    if not files:
        return 0

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    documents: list[Document] = []
    for file_path in files:
        if file_path.stat().st_size > max_file_size_kb * 1024:
            continue
        try:
            content = file_path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue

        chunks = splitter.split_text(content)
        for chunk in chunks:
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "source": str(file_path.relative_to(root)),
                        "file_path": str(file_path),
                    },
                )
            )

    if not documents:
        return 0

    try:
        embeddings = OllamaEmbeddings(model=embeddings_model)
        persist_dir.mkdir(parents=True, exist_ok=True)

        vectorstore = Chroma(
            collection_name="code-assist",
            embedding_function=embeddings,
            persist_directory=str(persist_dir),
        )
        vectorstore.add_documents(documents)
    except Exception as exc:  # pragma: no cover - surface friendly error
        message = (
            "Failed to reach Ollama at http://localhost:11434. "
            "Make sure the Ollama app is running and the embedding model is pulled."
        )
        raise RuntimeError(message) from exc
    return len(documents)


def has_vectorstore(persist_dir: Path) -> bool:
    if not persist_dir.exists():
        return False
    if (persist_dir / "chroma.sqlite3").exists():
        return True
    for path in persist_dir.rglob("*"):
        if path.is_file():
            return True
    return False


def _iter_files(
    root: Path,
    extensions: set[str],
    ignore_dirs: set[str],
) -> Iterable[Path]:
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if name not in ignore_dirs]
        for name in filenames:
            file_path = Path(dirpath) / name
            if file_path.suffix.lower() not in extensions:
                continue
            yield file_path
