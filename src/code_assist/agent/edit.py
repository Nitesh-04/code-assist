from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import ast
import difflib

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from code_assist.rag.retriever import query_repo


@dataclass(frozen=True)
class EditResult:
    updated_content: str
    diff: str
    syntax_error: str | None = None


_PLACEHOLDER_MARKERS = (
    "rest of the file",
    "rest of file",
    "...",
    "assuming",
    "if i were",
    "i would",
    "i will",
    "i'll",
)


def _has_placeholder_lines(content: str) -> bool:
    for line in content.splitlines():
        stripped = line.strip().lower()
        if not stripped:
            continue
        if stripped in {"...", "# ...", "# (rest)"}:
            return True
        if stripped.startswith("# ..."):
            return True
        if any(marker in stripped for marker in _PLACEHOLDER_MARKERS):
            return True
    return False


def _looks_truncated(original: str, updated: str) -> bool:
    original_lines = [line for line in original.splitlines() if line.strip()]
    updated_lines = [line for line in updated.splitlines() if line.strip()]
    if original_lines and len(updated_lines) < max(10, int(len(original_lines) * 0.6)):
        return True
    return _has_placeholder_lines(updated)


def _strip_wrappers(content: str) -> tuple[str, bool]:
    stripped = content.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 2 and lines[0].startswith("```") and lines[-1].startswith("```"):
            return "\n".join(lines[1:-1]).strip("\n"), True
    if stripped.startswith("'''" ) and stripped.endswith("'''" ) and stripped.count("'''") == 2:
        return stripped[3:-3].strip("\n"), True
    if stripped.startswith('"""') and stripped.endswith('"""') and stripped.count('"""') == 2:
        return stripped[3:-3].strip("\n"), True
    return content, False


def _validate_python_syntax(content: str, path: Path) -> str | None:
    try:
        ast.parse(content, filename=str(path))
    except SyntaxError as exc:
        return f"Invalid Python syntax: {exc.msg} at line {exc.lineno}"
    return None


def propose_edit(
    root: Path,
    path: Path,
    instruction: str,
    *,
    chat_model: str,
    embed_model: str,
    allow_rewrite: bool = False,
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

    llm = ChatOllama(model=chat_model)

    def _build_prompt(strict: bool) -> ChatPromptTemplate:
        suffix = (
            "Never include commentary, placeholders, or summaries. "
            "Do not say 'rest of the file' or use ellipses."
        )
        system = (
            "You are a local AI dev copilot. Update the file content based on the instruction. "
            "Add the new part to the current file content, do not remove any unchanged parts. "
            "Return ONLY the full updated file content. Do not include code fences. "
            "When editing Python, the result must be valid Python syntax. "
            + (suffix if strict else "")
        )
        user = (
            "Instruction: {instruction}\n\nFile path: {path}\n\n"
            "Requirements:\n- Preserve all unchanged code.\n- Output must be valid syntax for the file type.\n"
            "For Python files, validate syntax before responding.\n\n"
            "Current content:\n{content}\n\nRelevant context:\n{context}"
        )
        if strict:
            user += (
                "\n\nStrict output rules:\n"
                "- Return the entire file, not a snippet.\n"
                "- Do not include notes, explanations, or placeholder lines.\n"
            )
        return ChatPromptTemplate.from_messages([("system", system), ("user", user)])

    updated = ""
    for attempt in range(2):
        prompt = _build_prompt(strict=attempt == 1)
        message = prompt.format_messages(
            instruction=instruction,
            path=str(path),
            content=content,
            context="\n\n".join(context_lines),
        )
        response = llm.invoke(message)
        updated_raw = response.content or ""
        updated, _ = _strip_wrappers(updated_raw)
        if updated.strip() and not _looks_truncated(content, updated):
            break
    if not updated.strip():
        raise ValueError("Model returned empty content.")
    syntax_error = None
    if path.suffix.lower() == ".py":
        syntax_error = _validate_python_syntax(updated, path)

    similarity = difflib.SequenceMatcher(None, content, updated).ratio()
    if similarity < 0.2 and not allow_rewrite:
        raise ValueError(
            "Model response diverges too much from the original file. "
            "Re-run with a narrower instruction or use --allow-rewrite."
        )

    diff = "\n".join(
        difflib.unified_diff(
            content.splitlines(),
            updated.splitlines(),
            fromfile=str(path),
            tofile=str(path),
            lineterm="",
        )
    )
    return EditResult(updated_content=updated, diff=diff, syntax_error=syntax_error)
