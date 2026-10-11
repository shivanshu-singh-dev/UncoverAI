"""Lightweight `.env` file parser and environment configuration loader.

Loads key-value pairs from a local `.env` file into `os.environ` (without overwriting
already-exported shell variables by default) so the CLI, pipeline, and Streamlit UI
automatically use `.env` without manual shell export.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import MutableMapping, Optional

DEFAULT_DOTENV_PATH = Path(".env")
_DOTENV_LOADED: bool = False


def parse_dotenv_file(dotenv_path: Path = DEFAULT_DOTENV_PATH) -> dict[str, str]:
    """Parse a `.env` file and return a dictionary of key-value strings.

    Ignores blank lines and `#` comments. Strips optional surrounding single or double quotes.
    Never logs or prints secret values.
    """
    parsed: dict[str, str] = {}
    if not dotenv_path.is_file():
        return parsed

    try:
        content = dotenv_path.read_text(encoding="utf-8")
    except OSError:
        return parsed

    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue

        key, raw_val = line.split("=", 1)
        key = key.strip()
        if not key:
            continue

        val = raw_val.strip()
        # Strip inline comment if unquoted
        if val and val[0] not in ('"', "'") and " #" in val:
            val = val.split(" #", 1)[0].strip()

        if len(val) >= 2 and ((val[0] == '"' and val[-1] == '"') or (val[0] == "'" and val[-1] == "'")):
            val = val[1:-1]

        parsed[key] = val

    return parsed


def load_dotenv_if_present(
    dotenv_path: Path = DEFAULT_DOTENV_PATH,
    *,
    target_env: Optional[MutableMapping[str, str]] = None,
    override: bool = False,
) -> dict[str, str]:
    """Load key-value pairs from `.env` into `target_env` (defaults to `os.environ`).

    Returns the parsed key-value dictionary from `dotenv_path`.
    """
    global _DOTENV_LOADED
    parsed = parse_dotenv_file(dotenv_path)
    env_map = os.environ if target_env is None else target_env

    for k, v in parsed.items():
        if override or k not in env_map:
            env_map[k] = v

    if target_env is None and dotenv_path == DEFAULT_DOTENV_PATH:
        _DOTENV_LOADED = True
    return parsed
