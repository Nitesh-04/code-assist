from __future__ import annotations

from pathlib import Path
import shlex
import subprocess
from typing import Sequence


ALLOWED_COMMANDS = {
    "python",
    "pytest",
    "ruff",
    "uv",
}


def execute_command(command: str, root: Path, *, timeout: int = 20) -> str:
    parts = shlex.split(command)
    if not parts:
        raise ValueError("Command is empty")

    if parts[0] not in ALLOWED_COMMANDS:
        raise ValueError(f"Command not allowed: {parts[0]}")

    result = subprocess.run(
        parts,
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    output = result.stdout.strip()
    if result.stderr:
        output = f"{output}\n{result.stderr.strip()}".strip()
    return output or "(no output)"
