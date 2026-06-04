# Code Assist

Local-first AI dev copilot for your codebase.

## Quickstart

```bash
uv run code-assist hello
```

## CLI commands

```bash
code-assist hello
code-assist read path/to/file.py
code-assist search "query text"
code-assist exec "pytest -q"
code-assist chat
code-assist init-config
code-assist show-config

code-assist rag ingest
code-assist rag query "Where is the CLI defined?"

code-assist agent ask "Where is the CLI defined?"
code-assist agent where "tools"
code-assist agent explain-file src/code_assist/cli/main.py
code-assist agent summarize "RAG ingestion"
code-assist agent edit src/code_assist/cli/main.py "Add a new CLI command for status"
```

## RAG (local ChromaDB)

```bash
uv run code-assist rag ingest
uv run code-assist rag query "Where is the CLI defined?"
```

Set the embedding model used by Ollama:

```bash
export CODE_ASSIST_EMBED_MODEL=nomic-embed-text
```

Make sure Ollama is running and the embedding model is pulled:

```bash
ollama serve
ollama pull nomic-embed-text
```

## Agent (LangGraph)

```bash
uv run code-assist agent ask "Where is the CLI defined?"
```

Richer tasks:

```bash
uv run code-assist agent where "tools"
uv run code-assist agent explain-file src/code_assist/cli/main.py
uv run code-assist agent summarize "RAG ingestion"
```

Interactive session:

```bash
code-assist
code-assist chat
```

On first run in a repo, it will ingest the codebase and create
`.ai-copilot/vectorstore` automatically.

Edit a file (review diff before applying):

```bash
uv run code-assist agent edit src/code_assist/cli/main.py "Add a new CLI command for status"
```

Set the chat model used by Ollama:

```bash
export CODE_ASSIST_CHAT_MODEL=gemma4:e4b
```

## Model profiles

Create a repo-local config file to set model defaults:

```bash
uv run code-assist init-config
```

Edit `.ai-copilot/config.json`:

```json
{
	"models": {
		"chat": "gemma4:e4b",
		"embed": "nomic-embed-text"
	}
}
```

Show the resolved config:

```bash
uv run code-assist show-config
```
