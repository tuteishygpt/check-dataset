"""Gemini API Integration Module.

Handles interaction with Google Gemini API including:
- Synchronous audio transcription.
- Batch processing with file reuse caching.
- File management (upload/registry).
"""
from __future__ import annotations

import base64
import io
import json
import mimetypes
import os
import random
import re
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import requests
import soundfile as sf

# Try imports
try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None


REGISTRY_FILE = "gemini_file_registry.json"


def is_transcription_error(text: str) -> bool:
    """Check if the transcription result is an error message rather than real text.
    
    Error responses from transcribe_audio start with 'Error:'.
    """
    if not text:
        return False
    return text.startswith("Error:")


DEFAULT_TRANSCRIPTION_PROMPT = """You are a transcription engine.
Transcribe the following audio verbatim in Belarusian.
This audio is a fragment of an audiobook and may start or end mid-sentence.
Preserve exact wording, punctuation, repetitions, pauses, and incomplete or cut-off phrases.
Do NOT correct grammar, normalize text, or improve style.
Write all numbers as Belarusian words (no digits), preserving the intended form (cardinal/ordinal, cases, and gender when clear from context). If the form is unclear, choose the most neutral spoken form.
Do NOT add explanations, timestamps, speaker labels, or any extra text.
Output ONLY the raw transcription."""


@dataclass
class BatchTask:
    """Represents a single audio file queued for BATCH processing."""
    key: str
    path: str
    mime_type: str = "audio/wav"
    file_uri: Optional[str] = None


class GeminiFileRegistry:
    """Manages a local registry of files uploaded to Google Gemini."""

    def __init__(self, registry_path: str = REGISTRY_FILE):
        self.registry_path = registry_path
        self._registry: Dict[str, Dict[str, Any]] = self._load_registry()

    def _load_registry(self) -> Dict[str, Dict[str, Any]]:
        if os.path.exists(self.registry_path):
            try:
                with open(self.registry_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def save_registry(self):
        try:
            with open(self.registry_path, "w", encoding="utf-8") as f:
                json.dump(self._registry, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Warning: Failed to save file registry: {e}")

    def get_file(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Check if file is in registry and return its info."""
        abs_path = os.path.abspath(file_path)
        if abs_path in self._registry:
            entry = self._registry[abs_path]
            # TODO: Implement expiration check if needed (Gemini files expire in 48h)
            # For now, we assume if it's in registry, it might be valid.
            # Ideally we should store upload timestamp.
            return entry
        return None

    def add_file(self, file_path: str, uri: str, name: str, mime_type: str):
        """Add a file to the registry."""
        abs_path = os.path.abspath(file_path)
        self._registry[abs_path] = {
            "uri": uri,
            "name": name,
            "mime_type": mime_type,
            "upload_time": time.time(),
            "path": abs_path
        }
        self.save_registry()


class GeminiIntegrator:
    """Main Class for Gemini Integration."""

    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("API Key is required for GeminiIntegrator")
        if genai is None:
            raise RuntimeError("google-genai library is not installed")
        
        self.api_key = api_key
        self.client = genai.Client(api_key=api_key)
        self.file_registry = GeminiFileRegistry()

    # -------------------------------------------------------------------------
    # Synchronous Transcription (formerly in utils.transcribe_audio)
    # -------------------------------------------------------------------------
    def transcribe_audio(
        self,
        model_name: str,
        audio_array,
        sampling_rate,
        config=None,
        max_retries: int = 5,
        prompt: str = None
    ) -> str:
        """
        Transcribes audio using Gemini API (Sync).
        """
        # Convert numpy array to bytes (WAV format)
        audio_buffer = io.BytesIO()
        try:
            sr = int(float(sampling_rate)) if sampling_rate is not None else 16000
        except (ValueError, TypeError):
            sr = 16000
            
        sf.write(audio_buffer, audio_array, sr, format='WAV')
        audio_bytes = audio_buffer.getvalue()
        
        last_error = None
        
        final_prompt = prompt if prompt else DEFAULT_TRANSCRIPTION_PROMPT

        for attempt in range(max_retries):
            from core.state import get_stop_requested
            if get_stop_requested():
                return "Error: Cancelled by user"
            try:
                # Generate content
                response = self.client.models.generate_content(
                    model=model_name,
                    contents=[
                        types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav"),
                        final_prompt
                    ],
                    config=config
                )
                
                return response.text.strip()
                
            except Exception as e:
                error_str = str(e)
                last_error = e
                
                # Check for 429
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    # Parse retry delay
                    wait_time = 60
                    retry_match = re.search(r'retry in (\d+(?:\.\d+)?)s', error_str)
                    if retry_match:
                        wait_time = float(retry_match.group(1)) + random.uniform(1, 5)
                    else:
                        wait_time = 60 * (2 ** attempt) + random.uniform(1, 10)
                    
                    if attempt < max_retries - 1:
                        print(f"⏳ Retry {attempt + 1}/{max_retries}. Waiting {wait_time:.1f}s...")
                        time.sleep(wait_time)
                        continue
                else:
                    return f"Error: {e}"
        
        return f"Error: Max retries exceeded. Last error: {last_error}"

    # -------------------------------------------------------------------------
    # Batch Processing
    # -------------------------------------------------------------------------
    def run_batch(
        self,
        tasks: Iterable[BatchTask],
        model_name: str,
        prompt_text: str,
        chunk_size: int = 500,
    ) -> Dict[str, str]:
        """Run batch jobs and return mapping key -> text."""
        
        pending = list(tasks)
        if not pending:
            return {}

        results: Dict[str, str] = {}
        normalized_chunk_size = max(1, int(chunk_size))
        
        # Prepare content (upload files if needed)
        self._prepare_files_for_batch(pending)

        for chunk_idx in range(0, len(pending), normalized_chunk_size):
            from core.state import get_stop_requested
            if get_stop_requested():
                break
            chunk = pending[chunk_idx : chunk_idx + normalized_chunk_size]
            self._process_chunk(chunk, chunk_idx // normalized_chunk_size, model_name, prompt_text, results)

        return results

    def _prepare_files_for_batch(self, tasks: List[BatchTask]):
        """Uploads files if they are not already in the registry/cloud."""
        for task in tasks:
            from core.state import get_stop_requested
            if get_stop_requested():
                break
            # Check registry
            entry = self.file_registry.get_file(task.path)
            
            if entry:
                # Use existing URI
                task.file_uri = entry['uri']
                task.mime_type = entry['mime_type']  # Ensure mime match
                # Check if file is actually valid on server?
                # For now assume yes. If it fails, we might need logic to re-upload.
            else:
                # Upload
                print(f"Uploading {task.path}...")
                try:
                    uploaded = self.client.files.upload(file=task.path)
                    
                    # Store in registry
                    self.file_registry.add_file(
                        file_path=task.path,
                        uri=uploaded.uri,
                        name=uploaded.name,
                        mime_type=uploaded.mime_type or task.mime_type
                    )
                    
                    task.file_uri = uploaded.uri
                    task.mime_type = uploaded.mime_type or task.mime_type
                    print(f"Uploaded: {uploaded.uri}")
                except Exception as e:
                    print(f"Error uploading {task.path}: {e}")
                    # Mark task as failed or skip?
                    task.file_uri = None

    def _process_chunk(
        self, 
        chunk: List[BatchTask], 
        chunk_index: int,
        model_name: str, 
        prompt_text: str, 
        results: Dict[str, str]
    ) -> None:
        if not chunk:
            return

        valid_tasks = [t for t in chunk if t.file_uri]
        if not valid_tasks:
            for t in chunk:
                results[t.key] = "Error: File upload failed"
            return

        uploaded_jsonl_name: Optional[str] = None
        
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                jsonl_path = os.path.join(tmpdir, f"batch_input_{chunk_index:03}.jsonl")
                self._prepare_chunk_jsonl(valid_tasks, jsonl_path, prompt_text, chunk_index)
                
                # Upload JSONL input (this one is ephemeral)
                uploaded_jsonl = self.client.files.upload(
                    file=jsonl_path,
                    config=types.UploadFileConfig(
                        display_name=f"batch-input-{chunk_index:03}", 
                        mime_type="application/json"
                    )
                )
                uploaded_jsonl_name = uploaded_jsonl.name
                
                print(f"Batch {chunk_index}: JSONL uploaded {uploaded_jsonl_name}")

            # Create Batch Job
            batch_name = self._create_batch_job_rest(
                model_id=model_name,
                input_file_name=uploaded_jsonl_name,
                display_name=f"audio-batch-{chunk_index:03}",
            )
            
            print(f"Batch {chunk_index}: Job started {batch_name}. Polling...")
            
            dest_file_name = self._poll_batch_job(batch_name)
            
            print(f"Batch {chunk_index}: Downloading results from {dest_file_name}...")
            
            # Download results
            file_content = self.client.files.download(file=dest_file_name)
            self._process_results_jsonl_bytes(file_content, results)
            
        except Exception as e:
            print(f"Batch {chunk_index} Error: {e}")
            for t in valid_tasks:
                if t.key not in results:
                    results[t.key] = f"Error in batch: {e}"
        finally:
            # Cleanup JSONL file only
            if uploaded_jsonl_name:
                try:
                    self.client.files.delete(name=uploaded_jsonl_name)
                except Exception:
                    pass
            # We DO NOT delete the content files (audio) as they are registered for reuse

    # ------------------------- JSONL helpers -------------------------
    def _prepare_chunk_jsonl(
        self, tasks_chunk: List[BatchTask], jsonl_path: str, prompt_text: str, chunk_index: int
    ) -> None:
        os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)

        with open(jsonl_path, "w", encoding="utf-8") as f:
            for i, task in enumerate(tasks_chunk):
                unique_key = task.key or f"chunk{chunk_index:03}_batch_{i:03}"
                parts = self._build_parts_for_task(task, prompt_text)
                request_entry = {
                    "key": unique_key,
                    "request": {
                        "contents": [
                            {
                                "role": "user",
                                "parts": parts,
                            }
                        ]
                    },
                }
                f.write(json.dumps(request_entry, ensure_ascii=False) + "\n")

    @staticmethod
    def _build_parts_for_task(task: BatchTask, prompt_text: str) -> List[Dict[str, Any]]:
        clean_prompt = (prompt_text or "").strip()
        parts: List[Dict[str, Any]] = []
        if clean_prompt:
            parts.append({"text": clean_prompt})
        
        # Ensure mime_type is never None/Empty, fallback to audio/wav
        mime = task.mime_type if task.mime_type else "audio/wav"
        
        parts.append(
            {
                "file_data": {
                    "mime_type": mime,
                    "file_uri": task.file_uri,
                }
            }
        )
        return parts

    # ------------------------- REST helpers --------------------------
    # Note: Using REST for Batch Create because sometimes SDK can be finicky or user provided implementation used requests.
    # The provided gemini_batch.py used requests for creating batch job. I will keep that logic.
    
    @staticmethod
    def _rest_model_name(model_id: str) -> str:
        return model_id.replace("models/", "")

    def _create_batch_job_rest(self, model_id: str, input_file_name: str, display_name: str) -> str:
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self._rest_model_name(model_id)}:batchGenerateContent"
        )
        headers = {
            "x-goog-api-key": self.api_key,
            "Content-Type": "application/json",
        }
        payload = {
            "batch": {
                "display_name": display_name,
                "input_config": {"file_name": input_file_name},
            }
        }

        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        
        if not resp.ok:
             # Try SDK fallback if requests fails? Or just raise error.
             # Note: SDK support for Batch might be available via client.batches.create
             # But let's stick to the code user provided as baseline
            raise RuntimeError(f"REST create failed: {resp.status_code} {resp.text}")

        data = resp.json()
        name = data.get("name")
        if not name and isinstance(data.get("batch"), dict):
            name = data["batch"].get("name")
        if not name:
            raise RuntimeError(f"REST create succeeded but no batch name found. Response: {data}")

        return name

    def _get_batch_job_rest(self, name: str) -> Dict[str, Any]:
        url = f"https://generativelanguage.googleapis.com/v1beta/{name}"
        headers = {"x-goog-api-key": self.api_key}
        resp = requests.get(url, headers=headers, timeout=60)
        if not resp.ok:
            raise RuntimeError(f"REST get failed: {resp.status_code} {resp.text}")
        return resp.json()

    def _poll_batch_job(self, batch_name: str) -> str:
        completed_states = {
            "BATCH_STATE_SUCCEEDED",
            "BATCH_STATE_FAILED",
            "BATCH_STATE_CANCELLED",
            "BATCH_STATE_EXPIRED",
            "BATCH_STATE_PAUSED",
        }

        while True:
            from core.state import get_stop_requested
            if get_stop_requested():
                raise RuntimeError("Batch job polling cancelled by user")
            rest_job = self._get_batch_job_rest(batch_name)
            state = rest_job.get("state") or (rest_job.get("metadata") or {}).get("state") or (rest_job.get("batch") or {}).get("state")
            
            print(f"Job {batch_name} state: {state}")
            
            if state in completed_states:
                break
            time.sleep(30)

        if state != "BATCH_STATE_SUCCEEDED":
            err = rest_job.get("error") or (rest_job.get("response") or {}).get("error")
            raise RuntimeError(f"Batch job failed with state {state}: {err}")

        # Extract result file
        resp = rest_job.get("response") or {}
        dest = resp.get("dest") or {}
        result_file_name = (
            dest.get("file_name")
            or dest.get("fileName")
            or resp.get("file_name")
            or resp.get("fileName")
            or resp.get("responsesFile")
            or resp.get("responses_file")
        )
        
        if not result_file_name:
            # Sometimes it's nested differently in different API versions
            # Just dumping entire object for debug might be needed if this fails
            raise RuntimeError(f"Could not locate result file name in REST response: {rest_job}")
            
        return result_file_name

    def _process_results_jsonl_bytes(self, content_bytes: bytes, results: Dict[str, str]) -> None:
        content_str = content_bytes.decode("utf-8", errors="replace")

        for line in content_str.splitlines():
            if not line.strip():
                continue
            try:
                result = json.loads(line)
            except Exception:
                continue

            key = result.get("key")
            if not key:
                continue

            response_wrapper = result.get("response", {})
            if "error" in response_wrapper:
                results[key] = f"Error: {response_wrapper['error']}"
                continue

            candidates = response_wrapper.get("candidates", [])
            text: Optional[str] = None
            if candidates and "content" in candidates[0]:
                parts = candidates[0]["content"].get("parts", [])
                for part in parts:
                    if isinstance(part, dict) and part.get("text"):
                        text = part["text"]
                        break
            
            if text is None:
                text = "" # Empty transcription?

            results[key] = text
