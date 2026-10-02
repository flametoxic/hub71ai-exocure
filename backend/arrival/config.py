"""Environment-backed runtime configuration for the CURE backend."""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re


def load_environment(path: Path | None = None) -> None:
    location = path or Path(__file__).resolve().parents[1] / '.env'
    if not location.is_file():
        return
    for line in location.read_text(encoding='utf-8-sig').splitlines():
        name, separator, value = line.strip().partition('=')
        name, value = name.strip(), value.strip()
        if not separator or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', name):
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in '\"\'':
            value = value[1:-1]
        else:
            value = value.split(' #', 1)[0].rstrip()
        if value:
            os.environ.setdefault(name, value)


@dataclass(frozen=True)
class Settings:
    language_mode: str = "local"
    openai_api_key: str | None = None
    openai_model: str = "gpt-6.1-sol"
    openai_embed_model: str = "text-embedding-3-small"
    openai_agent_model: str | None = None
    reasoning_effort: str = "medium"
    cors_origins: tuple[str, ...] = ("http://localhost:3000", "http://localhost:5173")
    host: str = "0.0.0.0"
    port: int = 8000
    serve_web: bool = True

    @classmethod
    def from_env(cls) -> "Settings":
        load_environment()
        mode = os.environ.get("CURE_LANGUAGE_MODE", "auto").strip().lower()
        if mode == 'auto':
            mode = 'openai' if os.environ.get('OPENAI_API_KEY') else 'local'
        if mode not in {"local", "openai"}:
            raise ValueError("CURE_LANGUAGE_MODE must be 'auto', 'local' or 'openai'")
        raw_origins = os.environ.get("CURE_CORS_ORIGINS", "http://localhost:3000,http://localhost:5173")
        origins = tuple(origin.strip() for origin in raw_origins.split(",") if origin.strip())
        serve_web = os.environ.get("CURE_SERVE_WEB", "1").strip().lower() not in {"0", "false", "no"}
        return cls(
            language_mode=mode,
            openai_api_key=os.environ.get("OPENAI_API_KEY") or None,
            openai_model=os.environ.get("OPENAI_MODEL", os.environ.get("ARRIVAL_OPENAI_MODEL", "gpt-6.1-sol")),
            reasoning_effort=os.environ.get("OPENAI_REASONING_EFFORT", "medium"),
            openai_embed_model=os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-small"),
            openai_agent_model=os.environ.get('OPENAI_MODEL_AGENT') or None,
            cors_origins=origins,
            host=os.environ.get("HOST", "0.0.0.0"),
            port=int(os.environ.get("PORT", "8000")),
            serve_web=serve_web,
        )

    def language_ready(self) -> tuple[bool, str | None]:
        if self.language_mode == "openai" and not self.openai_api_key:
            return False, "OPENAI_API_KEY is required in openai mode"
        return True, None
