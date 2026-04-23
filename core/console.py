"""Console helpers that stay safe on legacy Windows encodings."""

from __future__ import annotations

import locale
import sys
from typing import TextIO


def _resolve_encoding(stream: TextIO) -> str:
    encoding = getattr(stream, "encoding", None)
    if encoding:
        return encoding
    return locale.getpreferredencoding(False) or "utf-8"


def _sanitize_for_stream(text: str, stream: TextIO) -> str:
    encoding = _resolve_encoding(stream)
    try:
        text.encode(encoding)
        return text
    except (LookupError, UnicodeEncodeError):
        return text.encode(encoding, errors="replace").decode(encoding, errors="replace")


def safe_print(*args, sep: str = " ", end: str = "\n", file: TextIO | None = None, flush: bool = False) -> None:
    """Print text without crashing on consoles that cannot encode Unicode emoji."""
    stream = file if file is not None else sys.stdout
    message = sep.join(str(arg) for arg in args)
    print(_sanitize_for_stream(message, stream), end=end, file=stream, flush=flush)
