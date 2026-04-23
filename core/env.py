"""Environment bootstrap helpers."""

from __future__ import annotations

import tempfile
from pathlib import Path

from dotenv import load_dotenv


def configure_environment(dotenv_path: str | None = None) -> None:
    """Load `.env` with override semantics and refresh tempdir resolution."""
    resolved_path = Path(dotenv_path) if dotenv_path else Path(__file__).resolve().parents[1] / ".env"
    load_dotenv(dotenv_path=resolved_path, override=True)
    tempfile.tempdir = None
