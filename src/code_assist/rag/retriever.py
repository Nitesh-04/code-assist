from __future__ import annotations

from pathlib import Path

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings


def query_repo(
    query: str,
    persist_dir: Path,
    embeddings_model: str,
    *,
    k: int = 4,
) -> list[dict[str, str]]:
    if not query.strip():
        return []

    try:
        embeddings = OllamaEmbeddings(model=embeddings_model)
        vectorstore = Chroma(
            collection_name="code-assist",
            embedding_function=embeddings,
            persist_directory=str(persist_dir),
        )
        results = vectorstore.similarity_search_with_score(query, k=k)
    except Exception as exc:  # pragma: no cover - surface friendly error
        message = (
            "Failed to reach Ollama at http://localhost:11434. "
            "Make sure the Ollama app is running and the embedding model is pulled."
        )
        raise RuntimeError(message) from exc
    output: list[dict[str, str]] = []
    for doc, score in results:
        output.append(
            {
                "source": doc.metadata.get("source", ""),
                "score": f"{score:.4f}",
                "content": doc.page_content,
            }
        )
    return output
