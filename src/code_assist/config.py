from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CHAT_MODEL = "qwen2.5:3b"
DEFAULT_EMBED_MODEL = "nomic-embed-text"


@dataclass(frozen=True)
class ModelConfig:
    chat_model: str = DEFAULT_CHAT_MODEL
    embed_model: str = DEFAULT_EMBED_MODEL


def load_model_config(root: Path) -> ModelConfig:
    config_path = root / ".ai-copilot" / "config.json"
    if not config_path.exists():
        return ModelConfig()

    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return ModelConfig()

    models = data.get("models", {})
    return ModelConfig(
        chat_model=models.get("chat", DEFAULT_CHAT_MODEL),
        embed_model=models.get("embed", DEFAULT_EMBED_MODEL),
    )


def ensure_config(root: Path) -> Path:
    config_path = root / ".ai-copilot" / "config.json"
    if config_path.exists():
        return config_path

    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(
            {
                "models": {
                    "chat": DEFAULT_CHAT_MODEL,
                    "embed": DEFAULT_EMBED_MODEL,
                }
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return config_path
