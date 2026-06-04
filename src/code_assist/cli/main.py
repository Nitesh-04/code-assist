from pathlib import Path
import shutil
import subprocess

import typer

from code_assist.agent.graph import AgentConfig, build_graph
from code_assist.agent.edit import propose_edit
from code_assist.config import ensure_config, load_model_config
from code_assist.memory import append_memory, format_memory, load_memory
from code_assist.rag.ingest import has_vectorstore, ingest_repo
from code_assist.rag.retriever import query_repo
from code_assist.tools.exec import execute_command
from code_assist.tools.files import read_file, search_code

app = typer.Typer(help="Local-first AI dev copilot for your codebase")
rag_app = typer.Typer(help="RAG commands")
agent_app = typer.Typer(help="Agent commands")
app.add_typer(rag_app, name="rag")
app.add_typer(agent_app, name="agent")


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    root: Path = Path.cwd(),
    chat_model: str | None = typer.Option(None, envvar="CODE_ASSIST_CHAT_MODEL"),
    embed_model: str | None = typer.Option(None, envvar="CODE_ASSIST_EMBED_MODEL"),
) -> None:
    """Local-first AI dev copilot for your codebase."""
    if ctx.invoked_subcommand is None:
        chat(root=root, chat_model=chat_model, embed_model=embed_model)


@app.command()
def hello() -> None:
    """Verify the CLI is wired correctly."""
    typer.echo("code-assist is ready")


@app.command()
def read(path: str, root: Path = Path.cwd()) -> None:
    """Read a file from the workspace root."""
    content = read_file(path, root)
    typer.echo(content)


@app.command("search")
def search_cmd(query: str, root: Path = Path.cwd(), max_results: int = 20) -> None:
    """Search for text in the workspace root."""
    matches = search_code(query, root, max_results=max_results)
    if not matches:
        typer.echo("No matches found")
        return
    typer.echo("\n".join(matches))


@app.command("exec")
def exec_cmd(command: str, root: Path = Path.cwd(), timeout: int = 20) -> None:
    """Execute an allowlisted command from the workspace root."""
    output = execute_command(command, root, timeout=timeout)
    typer.echo(output)


@rag_app.command("ingest")
def rag_ingest(
    root: Path = Path.cwd(),
    embed_model: str | None = typer.Option(None, envvar="CODE_ASSIST_EMBED_MODEL"),
    chunk_size: int = 800,
    chunk_overlap: int = 100,
) -> None:
    """Ingest the repo into a local ChromaDB store."""
    config = load_model_config(root)
    embed_model = embed_model or config.embed_model
    persist_dir = root / ".ai-copilot" / "vectorstore"
    count = ingest_repo(
        root,
        persist_dir,
        embeddings_model=embed_model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    typer.echo(f"Ingested {count} chunks into {persist_dir}")


@rag_app.command("query")
def rag_query(
    query: str,
    root: Path = Path.cwd(),
    embed_model: str | None = typer.Option(None, envvar="CODE_ASSIST_EMBED_MODEL"),
    k: int = 4,
) -> None:
    """Query the local ChromaDB store."""
    config = load_model_config(root)
    embed_model = embed_model or config.embed_model
    persist_dir = root / ".ai-copilot" / "vectorstore"
    results = query_repo(query, persist_dir, embeddings_model=embed_model, k=k)
    if not results:
        typer.echo("No results found")
        return
    for item in results:
        typer.echo(f"{item['source']} (score={item['score']})")
        typer.echo(item["content"])
        typer.echo("---")


@agent_app.command("ask")
def agent_ask(
    question: str,
    root: Path = Path.cwd(),
    chat_model: str | None = typer.Option(None, envvar="CODE_ASSIST_CHAT_MODEL"),
    embed_model: str | None = typer.Option(None, envvar="CODE_ASSIST_EMBED_MODEL"),
    show_observations: bool = False,
    auto_ingest: bool = True,
    status: bool = True,
) -> None:
    """Ask the local agent a question about the repo."""
    config = load_model_config(root)
    chat_model = chat_model or config.chat_model
    embed_model = embed_model or config.embed_model
    persist_dir = root / ".ai-copilot" / "vectorstore"
    if auto_ingest and not has_vectorstore(persist_dir):
        typer.echo("No vectorstore found. Ingesting repo...", err=True)
        ingest_repo(root, persist_dir, embeddings_model=embed_model)
    config = AgentConfig(chat_model=chat_model, embed_model=embed_model, root=root)
    graph = build_graph(config).compile()
    memory = format_memory(load_memory(root))
    if status:
        typer.echo("Thinking...", err=True)
    if status:
        last_action = None
        last_state = None
        try:
            for state in graph.stream(
                {"question": question, "memory": memory},
                stream_mode="values",
            ):
                last_state = state
                action = state.get("action")
                action_input = state.get("action_input", "")
                if action and action != last_action and action != "none":
                    if action == "search":
                        typer.echo(f"Searching: {action_input}", err=True)
                    elif action == "read":
                        typer.echo(f"Reading: {action_input}", err=True)
                    last_action = action
            state = last_state or {"answer": ""}
        except Exception:
            state = graph.invoke({"question": question, "memory": memory})
    else:
        state = graph.invoke({"question": question, "memory": memory})
    answer = state.get("answer", "")
    typer.echo(answer)
    append_memory(root, question, answer)
    if show_observations:
        observations = state.get("observations", [])
        if observations:
            typer.echo("\nObservations:")
            typer.echo("\n---\n".join(observations))


@agent_app.command("edit")
def agent_edit(
    path: str,
    instruction: str,
    root: Path = Path.cwd(),
    chat_model: str | None = typer.Option(None, envvar="CODE_ASSIST_CHAT_MODEL"),
    embed_model: str | None = typer.Option(None, envvar="CODE_ASSIST_EMBED_MODEL"),
    apply: bool = False,
    allow_rewrite: bool = False,
    open_diff: bool = True,
) -> None:
    """Propose edits for a file and optionally apply them."""
    config = load_model_config(root)
    chat_model = chat_model or config.chat_model
    embed_model = embed_model or config.embed_model
    try:
        result = propose_edit(
            root,
            Path(path),
            instruction,
            chat_model=chat_model,
            embed_model=embed_model,
            allow_rewrite=allow_rewrite,
        )
    except ValueError as exc:
        typer.echo(f"Edit rejected: {exc}", err=True)
        raise typer.Exit(code=1)
    typer.echo(result.diff or "(no changes)")
    if not result.diff:
        return
    edits_dir = root / ".ai-copilot" / "edits"
    edits_dir.mkdir(parents=True, exist_ok=True)
    safe_name = Path(path).as_posix().replace("/", "__") + ".diff"
    diff_path = edits_dir / safe_name
    diff_path.write_text(result.diff, encoding="utf-8")
    typer.echo(f"Diff saved to {diff_path}")
    updated_path = diff_path.with_suffix(".updated")
    updated_path.write_text(result.updated_content, encoding="utf-8")
    if open_diff:
        code_bin = shutil.which("code")
        if code_bin:
            subprocess.run(
                [code_bin, "--diff", str(root / path), str(updated_path)],
                check=False,
            )
        else:
            typer.echo(
                "VS Code CLI not found. Install it via 'Shell Command: Install code command in PATH'.",
                err=True,
            )
    if result.syntax_error:
        typer.echo(f"Edit warning: {result.syntax_error}", err=True)
        typer.echo("Not applying changes due to syntax error.", err=True)
        return
    if apply or typer.confirm("Apply these changes?"):
        (root / path).write_text(result.updated_content, encoding="utf-8")
        typer.echo("Applied changes")


@app.command("chat")
def chat(
    root: Path = Path.cwd(),
    chat_model: str | None = typer.Option(None, envvar="CODE_ASSIST_CHAT_MODEL"),
    embed_model: str | None = typer.Option(None, envvar="CODE_ASSIST_EMBED_MODEL"),
) -> None:
    """Start an interactive chat session."""
    typer.echo("Interactive session. Type 'exit' to quit.")
    config = load_model_config(root)
    embed_model = embed_model or config.embed_model
    persist_dir = root / ".ai-copilot" / "vectorstore"
    if not has_vectorstore(persist_dir):
        typer.echo("No vectorstore found. Ingesting repo...", err=True)
        ingest_repo(root, persist_dir, embeddings_model=embed_model)
    while True:
        question = typer.prompt("You")
        if question.strip().lower() in {"exit", "quit"}:
            break
        agent_ask(
            question,
            root=root,
            chat_model=chat_model,
            embed_model=embed_model,
            auto_ingest=False,
        )


@agent_app.command("where")
def agent_where(
    query: str,
    root: Path = Path.cwd(),
    chat_model: str | None = typer.Option(None, envvar="CODE_ASSIST_CHAT_MODEL"),
    embed_model: str | None = typer.Option(None, envvar="CODE_ASSIST_EMBED_MODEL"),
) -> None:
    """Find where something is defined in the repo."""
    question = f"Where is {query} defined?"
    agent_ask(question, root=root, chat_model=chat_model, embed_model=embed_model)


@agent_app.command("explain-file")
def agent_explain_file(
    path: str,
    root: Path = Path.cwd(),
    chat_model: str | None = typer.Option(None, envvar="CODE_ASSIST_CHAT_MODEL"),
    embed_model: str | None = typer.Option(None, envvar="CODE_ASSIST_EMBED_MODEL"),
) -> None:
    """Explain a file and its role in the repo."""
    question = f"Explain the file at {path} and what it does."
    agent_ask(question, root=root, chat_model=chat_model, embed_model=embed_model)


@agent_app.command("summarize")
def agent_summarize(
    topic: str,
    root: Path = Path.cwd(),
    chat_model: str | None = typer.Option(None, envvar="CODE_ASSIST_CHAT_MODEL"),
    embed_model: str | None = typer.Option(None, envvar="CODE_ASSIST_EMBED_MODEL"),
) -> None:
    """Summarize a topic using repo context."""
    question = f"Summarize {topic} in this repo."
    agent_ask(question, root=root, chat_model=chat_model, embed_model=embed_model)


@app.command("init-config")
def init_config(root: Path = Path.cwd()) -> None:
    """Create a default .ai-copilot/config.json if missing."""
    config_path = ensure_config(root)
    typer.echo(f"Wrote {config_path}")


@app.command("show-config")
def show_config(root: Path = Path.cwd()) -> None:
    """Print the resolved model configuration."""
    config = load_model_config(root)
    typer.echo(f"chat={config.chat_model}")
    typer.echo(f"embed={config.embed_model}")
