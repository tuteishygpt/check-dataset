"""Vertex AI integration helpers for Gemini transcription workflows."""
from __future__ import annotations

import io
import json
import os
import random
import re
import time
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import soundfile as sf

try:
    from google import genai
    from google.genai import types
except ImportError:  # pragma: no cover - optional dependency wiring
    genai = None
    types = None

try:
    from google.cloud import storage
except ImportError:  # pragma: no cover - optional dependency wiring
    storage = None


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
VERTEX_BATCH_SIZE = 100
VERTEX_BATCH_POLL_INTERVAL_SECONDS = 30
VERTEX_BATCH_GCS_URI_ENV = "VERTEX_BATCH_GCS_URI"
FLEX_SUPPORTED_MODELS = (
    "gemini-3.1-flash-lite-preview",
    "gemini-3.1-flash-image-preview",
    "gemini-3.1-pro-preview",
    "gemini-3-flash-preview",
    "gemini-3-pro-image-preview",
)
BATCH_SUPPORTED_MODELS = (
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
)
FLEX_REQUEST_HEADERS = {
    "X-Vertex-AI-LLM-Request-Type": "shared",
    "X-Vertex-AI-LLM-Shared-Request-Type": "flex",
}
BATCH_COMPLETED_STATES = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_PAUSED",
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


def validate_batch_inference(
    model_name: str,
    *,
    staging_gcs_uri: Optional[str] = None,
) -> str:
    """Validate Vertex batch prerequisites and return the normalized staging prefix."""
    normalized_uri = normalize_gcs_uri_prefix(staging_gcs_uri or os.getenv(VERTEX_BATCH_GCS_URI_ENV))
    if not normalized_uri:
        raise RuntimeError(
            f"Vertex Batch mode requires {VERTEX_BATCH_GCS_URI_ENV}=gs://bucket/prefix."
        )
    if model_name not in BATCH_SUPPORTED_MODELS:
        supported_models = ", ".join(BATCH_SUPPORTED_MODELS)
        raise ValueError(
            "Vertex Batch mode currently supports only these models in this app: "
            f"{supported_models}. Received: {model_name}."
        )
    return normalized_uri


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


def chunk_by_size(items: Iterable[Any], chunk_size: int) -> List[List[Any]]:
    """Split items into fixed-size chunks."""
    chunk_size = max(1, int(chunk_size))
    sequence = list(items)
    return [sequence[index:index + chunk_size] for index in range(0, len(sequence), chunk_size)]


def normalize_gcs_uri_prefix(uri: Optional[str]) -> Optional[str]:
    """Normalize a GCS prefix without a trailing slash."""
    if not uri:
        return None
    normalized = str(uri).strip()
    if not normalized:
        return None
    if not normalized.startswith("gs://"):
        raise ValueError(f"Expected gs:// URI, received: {uri!r}")
    return normalized.rstrip("/")


def parse_gcs_uri(uri: str) -> Tuple[str, str]:
    """Split a gs:// URI into bucket and blob path."""
    normalized = normalize_gcs_uri_prefix(uri)
    if normalized is None:
        raise ValueError("GCS URI is required")
    without_scheme = normalized[5:]
    bucket_name, _, blob_name = without_scheme.partition("/")
    if not bucket_name:
        raise ValueError(f"Invalid GCS URI: {uri!r}")
    return bucket_name, blob_name


def sanitize_gcs_name(value: str) -> str:
    """Make a stable ASCII-ish suffix for GCS object names."""
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value or "").strip("._")
    return sanitized or "audio"


def build_batch_generation_config(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Convert direct SDK config into batch-compatible generationConfig."""
    source = dict(config or {})
    batch_config: Dict[str, Any] = {}
    if "temperature" in source:
        batch_config["temperature"] = float(source["temperature"])
    thinking_config = source.get("thinking_config") or {}
    if thinking_config:
        batch_config["thinkingConfig"] = {
            "includeThoughts": bool(thinking_config.get("include_thoughts", False)),
            "budgetTokens": int(thinking_config.get("budget_tokens", 0)),
        }
    return batch_config


def build_vertex_batch_request(
    *,
    audio_gcs_uri: str,
    mime_type: str,
    prompt: str,
    generation_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build one JSONL request row for Vertex batch inference."""
    return {
        "request": {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": prompt},
                        {
                            "fileData": {
                                "fileUri": audio_gcs_uri,
                                "mimeType": mime_type,
                            }
                        },
                    ],
                }
            ],
            "generationConfig": dict(generation_config or {}),
        }
    }


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


def _extract_request_file_uri(prediction: Dict[str, Any]) -> Optional[str]:
    request = prediction.get("request") or {}
    for content in request.get("contents", []) or []:
        for part in content.get("parts", []) or []:
            file_data = part.get("fileData") or {}
            file_uri = file_data.get("fileUri")
            if file_uri:
                return file_uri
    return None


def extract_text_from_batch_prediction(prediction: Dict[str, Any]) -> str:
    """Extract plain text from a Vertex batch output row."""
    response = prediction.get("response") or {}
    candidates = response.get("candidates") or []
    if not candidates:
        status = prediction.get("status")
        if status:
            return f"Error: {status}"
        return ""

    parts = ((candidates[0].get("content") or {}).get("parts") or [])
    text_fragments = [part.get("text", "") for part in parts if part.get("text")]
    return "".join(text_fragments).strip()


def normalize_batch_job_state(state: Any) -> str:
    """Normalize SDK enum/string state into the raw Vertex job state name."""
    if hasattr(state, "value"):
        return str(state.value)
    state_text = str(state)
    if "." in state_text:
        return state_text.split(".")[-1]
    return state_text


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
        self._storage_client = None

    def _ensure_storage_client(self):
        """Lazily initialize a GCS client for batch staging."""
        if storage is None:
            raise RuntimeError(
                "google-cloud-storage is not installed. Add it to requirements for Vertex Batch mode."
            )
        if self._storage_client is None:
            self._storage_client = storage.Client(project=self.project)
        return self._storage_client

    def _upload_bytes_to_gcs(self, destination_uri: str, payload: bytes, content_type: str) -> str:
        client = self._ensure_storage_client()
        bucket_name, blob_name = parse_gcs_uri(destination_uri)
        blob = client.bucket(bucket_name).blob(blob_name)
        blob.upload_from_string(payload, content_type=content_type)
        return destination_uri

    def _upload_text_to_gcs(self, destination_uri: str, payload: str, content_type: str) -> str:
        return self._upload_bytes_to_gcs(destination_uri, payload.encode("utf-8"), content_type)

    def _load_jsonl_from_gcs_prefix(self, prefix_uri: str) -> List[Dict[str, Any]]:
        client = self._ensure_storage_client()
        bucket_name, blob_prefix = parse_gcs_uri(prefix_uri)
        rows: List[Dict[str, Any]] = []
        for blob in client.list_blobs(bucket_name, prefix=blob_prefix):
            if not blob.name.endswith(".jsonl"):
                continue
            content = blob.download_as_text()
            for line in content.splitlines():
                stripped = line.strip()
                if not stripped:
                    continue
                rows.append(json.loads(stripped))
        return rows

    def _build_batch_run_prefix(self, staging_gcs_uri: str, model_name: str) -> str:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        suffix = sanitize_gcs_name(model_name)
        return f"{staging_gcs_uri}/{timestamp}_{suffix}_{random.randint(1000, 9999)}"

    def _stage_batch_audio(
        self,
        run_prefix: str,
        records: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        staged_records: List[Dict[str, Any]] = []
        total_records = max(1, len(records))
        for index, record in enumerate(records):
            from core.state import get_stop_requested

            if get_stop_requested():
                raise RuntimeError("Cancelled by user")

            record_id = int(record["id"])
            source_name = sanitize_gcs_name(os.path.basename(record.get("path") or f"record_{record_id}"))
            audio_uri = f"{run_prefix}/audio/{record_id:05d}_{source_name}.wav"
            audio_bytes = encode_audio_to_wav_bytes(record["audio_array"], record["sampling_rate"])
            self._upload_bytes_to_gcs(audio_uri, audio_bytes, "audio/wav")

            staged_record = dict(record)
            staged_record["audio_gcs_uri"] = audio_uri
            staged_records.append(staged_record)

            if (index + 1) % 20 == 0 or index + 1 == total_records:
                print(f"Uploaded {index + 1}/{total_records} audio files to GCS for Vertex Batch.")

        return staged_records

    def _create_batch_jobs(
        self,
        *,
        run_prefix: str,
        model_name: str,
        records: List[Dict[str, Any]],
        generation_config: Dict[str, Any],
        prompt: str,
    ) -> List[Dict[str, Any]]:
        jobs: List[Dict[str, Any]] = []
        for batch_index, chunk in enumerate(chunk_by_size(records, VERTEX_BATCH_SIZE)):
            input_uri = f"{run_prefix}/inputs/batch_{batch_index:03d}.jsonl"
            output_uri = f"{run_prefix}/outputs/batch_{batch_index:03d}"
            request_rows = [
                build_vertex_batch_request(
                    audio_gcs_uri=item["audio_gcs_uri"],
                    mime_type="audio/wav",
                    prompt=prompt,
                    generation_config=generation_config,
                )
                for item in chunk
            ]
            jsonl_payload = "\n".join(json.dumps(row, ensure_ascii=False) for row in request_rows)
            self._upload_text_to_gcs(input_uri, jsonl_payload, "application/jsonl")

            job = self.client.batches.create(
                model=model_name,
                src=input_uri,
                config={
                    "display_name": f"tts-validator-batch-{batch_index:03d}",
                    "dest": output_uri,
                },
            )
            jobs.append(
                {
                    "name": job.name,
                    "output_uri": output_uri,
                    "records": chunk,
                }
            )
        return jobs

    def _wait_for_batch_jobs(
        self,
        jobs: List[Dict[str, Any]],
        *,
        poll_interval_seconds: int = VERTEX_BATCH_POLL_INTERVAL_SECONDS,
    ) -> List[Dict[str, Any]]:
        remaining = {job["name"]: dict(job) for job in jobs}
        completed_jobs: List[Dict[str, Any]] = []

        while remaining:
            from core.state import get_stop_requested

            if get_stop_requested():
                raise RuntimeError("Cancelled by user")

            for job_name in list(remaining):
                job_info = remaining[job_name]
                batch_job = self.client.batches.get(name=job_name)
                state = normalize_batch_job_state(batch_job.state)
                if state not in BATCH_COMPLETED_STATES:
                    continue

                job_info["state"] = state
                job_info["job"] = batch_job
                completed_jobs.append(job_info)
                del remaining[job_name]

            if remaining:
                time.sleep(max(1, int(poll_interval_seconds)))

        return completed_jobs

    def transcribe_audio_batch(
        self,
        model_name: str,
        records: List[Dict[str, Any]],
        *,
        config: Optional[Dict[str, Any]] = None,
        prompt: Optional[str] = None,
        staging_gcs_uri: Optional[str] = None,
    ) -> Dict[int, str]:
        """Transcribe many audio records with Vertex Batch API via GCS staging."""
        if not records:
            return {}

        normalized_staging_uri = validate_batch_inference(
            model_name,
            staging_gcs_uri=staging_gcs_uri,
        )
        run_prefix = self._build_batch_run_prefix(normalized_staging_uri, model_name)
        final_prompt = prompt or DEFAULT_TRANSCRIPTION_PROMPT
        generation_config = build_batch_generation_config(config)

        staged_records = self._stage_batch_audio(run_prefix, records)
        jobs = self._create_batch_jobs(
            run_prefix=run_prefix,
            model_name=model_name,
            records=staged_records,
            generation_config=generation_config,
            prompt=final_prompt,
        )
        completed_jobs = self._wait_for_batch_jobs(jobs)

        outputs: Dict[int, str] = {}
        uri_to_record_id = {record["audio_gcs_uri"]: int(record["id"]) for record in staged_records}

        for job_info in completed_jobs:
            state = job_info["state"]
            if state != "JOB_STATE_SUCCEEDED":
                for record in job_info["records"]:
                    outputs[int(record["id"])] = f"Error: Batch job finished with {state}"
                continue

            rows = self._load_jsonl_from_gcs_prefix(job_info["output_uri"])
            for row in rows:
                record_uri = _extract_request_file_uri(row)
                record_id = uri_to_record_id.get(record_uri or "")
                if record_id is None:
                    continue
                if row.get("status"):
                    outputs[record_id] = f"Error: {row['status']}"
                    continue
                outputs[record_id] = extract_text_from_batch_prediction(row)

            for record in job_info["records"]:
                record_id = int(record["id"])
                outputs.setdefault(record_id, "Error: Missing batch output")

        return outputs

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
