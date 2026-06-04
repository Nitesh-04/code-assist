from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import difflib

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from code_assist.rag.retriever import query_repo


@dataclass(frozen=True)
class EditResult:
    updated_content: str
    diff: str


def propose_edit(
    root: Path,
    path: Path,
    instruction: str,
    *,
    chat_model: str,
    embed_model: str,
) -> EditResult:
    file_path = (root / path).resolve()
    if not file_path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    content = file_path.read_text(encoding="utf-8")
    retrieved = query_repo(
        instruction,
        root / ".ai-copilot" / "vectorstore",
        embeddings_model=embed_model,
    )
    context_lines = []
    for item in retrieved:
        context_lines.append(
            f"SOURCE: {item['source']}\nSCORE: {item['score']}\n{item['content']}"
        )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a local AI dev copilot. Update the file content based on the instruction. "
                "Return ONLY the full updated file content. Do not include code fences.",
            ),
            (
                "user",
                "Instruction: {instruction}\n\nFile path: {path}\n\nCurrent content:\n{content}\n\nRelevant context:\n{context}",
            ),
        ]
    )
    llm = ChatOllama(model=chat_model)
    message = prompt.format_messages(
        instruction=instruction,
        path=str(path),
        content=content,
        context="\n\n".join(context_lines),
    )
    response = llm.invoke(message)
    updated = response.content

    diff = "\n".join(
        difflib.unified_diff(
            content.splitlines(),
            updated.splitlines(),
            fromfile=str(path),
            tofile=str(path),
            lineterm="",
        )
    )
    return EditResult(updated_content=updated, diff=diff)
