"""Vertex AI integration helpers for Gemini transcription workflows."""
from __future__ import annotations

import io
import os
import random
import re
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import soundfile as sf

try:
    from google import genai
    from google.genai import types
except ImportError:  # pragma: no cover - optional dependency wiring
    genai = None
    types = None


DEFAULT_TRANSCRIPTION_PROMPT = """You are a transcription engine.
Transcribe the following audio verbatim in Belarusian.
This audio is a fragment of an audiobook and may start or end mid-sentence.
Preserve exact wording, punctuation, repetitions, pauses, and incomplete or cut-off phrases.
Do NOT correct grammar, normalize text, or improve style.
Write all numbers as Belarusian words (no digits), preserving the intended form (cardinal/ordinal, cases, and gender when clear from context). If the form is unclear, choose the most neutral spoken form.
Do NOT add explanations, timestamps, speaker labels, or any extra text.
Output ONLY the raw transcription."""

STANDARD_TIMEOUT_MS = 180000
FLEX_TIMEOUT_MS = 1800000
DEFAULT_VERTEX_LOCATION = "global"
FLEX_SUPPORTED_MODELS = (
    "gemini-3.1-flash-lite-preview",
    "gemini-3.1-flash-image-preview",
    "gemini-3.1-pro-preview",
    "gemini-3-flash-preview",
    "gemini-3-pro-image-preview",
)
FLEX_REQUEST_HEADERS = {
    "X-Vertex-AI-LLM-Request-Type": "shared",
    "X-Vertex-AI-LLM-Shared-Request-Type": "flex",
}


def is_transcription_error(text: str) -> bool:
    """Check if the transcription result is an error message rather than real text."""
    return bool(text) and text.startswith("Error:")


def supports_flex_inference(model_name: str, location: Optional[str] = None) -> bool:
    """Return whether the model/location pair supports Vertex Flex PayGo."""
    normalized_location = (location or os.getenv("GOOGLE_CLOUD_LOCATION") or DEFAULT_VERTEX_LOCATION).strip().lower()
    return normalized_location == DEFAULT_VERTEX_LOCATION and model_name in FLEX_SUPPORTED_MODELS


def validate_flex_inference(model_name: str, location: Optional[str] = None) -> None:
    """Validate the current model/location against Vertex Flex PayGo limits."""
    normalized_location = (location or os.getenv("GOOGLE_CLOUD_LOCATION") or DEFAULT_VERTEX_LOCATION).strip().lower()
    if normalized_location != DEFAULT_VERTEX_LOCATION:
        raise ValueError(
            f"Flex PayGo requires GOOGLE_CLOUD_LOCATION={DEFAULT_VERTEX_LOCATION}. "
            f"Current location: {location!r}."
        )

    if model_name not in FLEX_SUPPORTED_MODELS:
        supported_models = ", ".join(FLEX_SUPPORTED_MODELS)
        raise ValueError(
            "Flex PayGo supports only these preview models on Vertex AI: "
            f"{supported_models}. Received: {model_name}."
        )


def validate_vertex_environment(
    project: Optional[str] = None,
    location: Optional[str] = None,
) -> Tuple[str, str]:
    """Resolve and validate the Vertex AI project/location configuration."""
    resolved_project = project or os.getenv("GOOGLE_CLOUD_PROJECT")
    resolved_location = location or os.getenv("GOOGLE_CLOUD_LOCATION") or DEFAULT_VERTEX_LOCATION

    if not resolved_project:
        raise RuntimeError(
            "Vertex AI requires GOOGLE_CLOUD_PROJECT (and ADC via `gcloud auth application-default login`)."
        )

    return resolved_project, resolved_location


def build_generation_config(
    temperature: float,
    thinking_budget: int = 0,
    *,
    flex_mode: bool = False,
    timeout_ms: Optional[int] = None,
) -> Dict[str, Any]:
    """Build a generate_content config compatible with the Python GenAI SDK."""
    http_options: Dict[str, Any] = {
        "timeout": timeout_ms or (FLEX_TIMEOUT_MS if flex_mode else STANDARD_TIMEOUT_MS),
    }
    if flex_mode:
        http_options["api_version"] = "v1"
        http_options["headers"] = dict(FLEX_REQUEST_HEADERS)

    config: Dict[str, Any] = {
        "temperature": float(temperature),
        "http_options": http_options,
    }

    if thinking_budget and int(thinking_budget) > 0:
        config["thinking_config"] = {
            "include_thoughts": True,
            "budget_tokens": int(thinking_budget),
        }

    return config


def encode_audio_to_wav_bytes(audio_array, sampling_rate) -> bytes:
    """Re-encode an audio array into WAV bytes for inline Vertex requests."""
    audio_buffer = io.BytesIO()
    try:
        sample_rate = int(float(sampling_rate)) if sampling_rate is not None else 16000
    except (ValueError, TypeError):
        sample_rate = 16000

    sf.write(audio_buffer, np.asarray(audio_array), sample_rate, format="WAV")
    return audio_buffer.getvalue()


def _retry_delay_seconds(error_text: str, attempt: int) -> float:
    retry_match = re.search(r"retry in (\d+(?:\.\d+)?)s", error_text, flags=re.IGNORECASE)
    if retry_match:
        return float(retry_match.group(1)) + random.uniform(1, 3)
    return min(120.0, (2 ** attempt) * 5 + random.uniform(0.5, 2.0))


def _is_retryable_error(error_text: str) -> bool:
    upper_error = error_text.upper()
    return any(
        marker in upper_error
        for marker in ("429", "503", "RESOURCE_EXHAUSTED", "UNAVAILABLE", "DEADLINE_EXCEEDED")
    )


class GeminiIntegrator:
    """Compatibility wrapper that now talks to Gemini on Vertex AI."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        project: Optional[str] = None,
        location: Optional[str] = None,
    ):
        del api_key  # Gemini Developer API keys are no longer used in this project.

        if genai is None or types is None:
            raise RuntimeError("google-genai library is not installed")

        resolved_project, resolved_location = validate_vertex_environment(project=project, location=location)
        self.project = resolved_project
        self.location = resolved_location
        self.client = genai.Client(
            vertexai=True,
            project=resolved_project,
            location=resolved_location,
        )

    def transcribe_audio(
        self,
        model_name: str,
        audio_array,
        sampling_rate,
        config: Optional[Dict[str, Any]] = None,
        max_retries: int = 5,
        prompt: Optional[str] = None,
    ) -> str:
        """Transcribe audio through Vertex AI with inline audio bytes."""
        audio_bytes = encode_audio_to_wav_bytes(audio_array, sampling_rate)
        final_prompt = prompt or DEFAULT_TRANSCRIPTION_PROMPT
        final_config = dict(config or {})
        last_error: Optional[Exception] = None

        for attempt in range(max_retries):
            from core.state import get_stop_requested

            if get_stop_requested():
                return "Error: Cancelled by user"

            try:
                response = self.client.models.generate_content(
                    model=model_name,
                    contents=[
                        final_prompt,
                        types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav"),
                    ],
                    config=final_config,
                )
                return (response.text or "").strip()
            except Exception as exc:  # pragma: no cover - network behavior exercised via integration
                last_error = exc
                error_text = str(exc)
                if not _is_retryable_error(error_text) or attempt >= max_retries - 1:
                    return f"Error: {exc}"

                wait_time = _retry_delay_seconds(error_text, attempt)
                print(f"Retry {attempt + 1}/{max_retries} after {wait_time:.1f}s: {exc}")
                time.sleep(wait_time)

        return f"Error: Max retries exceeded. Last error: {last_error}"
