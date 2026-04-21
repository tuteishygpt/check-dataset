import io
import os
import random
import re
import time

from datasets import Audio, load_dataset
import librosa
import numpy as np
from rapidfuzz import fuzz
import soundfile as sf
from hf_auth import normalize_hf_token


def get_record_text(item):
    """Extract reference text from a dataset record."""
    return item.get("sentence") or item.get("text") or item.get("transcription") or item.get("transcript") or ""


def get_audio_path(item):
    """Extract the audio path from a dataset record."""
    audio_info = item.get("audio", {})
    if isinstance(audio_info, dict):
        return audio_info.get("path", "unknown")
    return "unknown"


def decode_audio_item(item):
    """Decode audio for a single dataset record on demand."""
    audio_info = item.get("audio", {})
    audio_path = get_audio_path(item)

    if not isinstance(audio_info, dict):
        return np.array([]), 16000, audio_path

    audio_array = audio_info.get("array")
    sampling_rate = audio_info.get("sampling_rate")
    if audio_array is not None and sampling_rate is not None:
        return audio_array, sampling_rate, audio_path

    audio_bytes = audio_info.get("bytes")
    if not audio_bytes:
        return np.array([]), 16000, audio_path

    audio_buffer = io.BytesIO(audio_bytes)
    audio_array, sampling_rate = librosa.load(audio_buffer, sr=None)
    return audio_array, sampling_rate, audio_path


def build_dataset_path_index(ds):
    """Build a lightweight lookup from full path/basename to dataset index."""
    path_index = {}
    for idx, item in enumerate(ds):
        audio_path = get_audio_path(item)
        if not audio_path or audio_path == "unknown":
            continue
        path_index[audio_path] = idx
        path_index[os.path.basename(audio_path)] = idx
    return path_index


def get_dataset_item(ds, result, path_index=None):
    """Resolve a dataset record for an analysis result."""
    source_idx = result.get("source_idx")
    if source_idx is not None:
        try:
            source_idx = int(source_idx)
            if 0 <= source_idx < len(ds):
                return ds[source_idx]
        except (TypeError, ValueError):
            pass

    path = result.get("path", "")
    if path_index is not None and path:
        lookup_idx = path_index.get(path) or path_index.get(os.path.basename(path))
        if lookup_idx is not None and 0 <= lookup_idx < len(ds):
            return ds[lookup_idx]

    record_id = result.get("id")
    if record_id is not None:
        try:
            record_id = int(record_id)
            if 0 <= record_id < len(ds):
                return ds[record_id]
        except (TypeError, ValueError):
            pass

    return None


def infer_result_dataset_limit(results):
    """Infer the smallest dataset subset that still covers current results."""
    if not results:
        return None

    source_indices = []
    for result in results:
        if not result:
            continue
        source_idx = result.get("source_idx")
        if source_idx is None:
            continue
        try:
            source_indices.append(int(source_idx))
        except (TypeError, ValueError):
            continue

    if source_indices:
        return max(source_indices) + 1

    return len(results)


def load_hf_dataset(
    dataset_name,
    split="train",
    limit=None,
    allowed_paths=None,
    hf_token=None,
    decode_audio=True,
):
    """
    Load a dataset from Hugging Face.

    By default the function preserves the old eager-decoding behavior for
    existing call sites. Pass ``decode_audio=False`` to keep audio in raw-bytes
    form and decode each sample only when it is needed.
    """
    try:
        hf_token = normalize_hf_token(hf_token)
        ds = load_dataset(dataset_name, split=split, token=hf_token)

        if "audio" in ds.features:
            ds = ds.cast_column("audio", Audio(decode=False))

        unique_items_map = {}
        allowed_lookup = set(allowed_paths) if allowed_paths is not None else None
        total_ds_count = len(ds)
        print(f"Loaded metadata records: {total_ds_count}")

        for item_idx, item in enumerate(ds):
            audio_path = get_audio_path(item)
            file_name = os.path.basename(audio_path)

            if allowed_lookup is not None and audio_path not in allowed_lookup and file_name not in allowed_lookup:
                continue

            if audio_path == "unknown":
                unique_items_map[f"unknown_{len(unique_items_map)}"] = item_idx
            else:
                unique_items_map[file_name] = item_idx

        selected_indices = list(unique_items_map.values())
        print(f"Unique files after dedupe: {len(selected_indices)}")

        if limit and limit > 0:
            selected_indices = selected_indices[:limit]
            print(f"Applied limit: {len(selected_indices)} files")

        selected_ds = ds.select(selected_indices)

        if not decode_audio:
            print(f"Ready for processing: {len(selected_ds)} files")
            return selected_ds

        processed_items = []
        for item in selected_ds:
            processed_item = dict(item)
            audio_array, sampling_rate, audio_path = decode_audio_item(item)
            processed_item["audio"] = {
                "array": audio_array,
                "sampling_rate": sampling_rate,
                "path": audio_path,
            }
            processed_items.append(processed_item)

        print(f"Ready for processing: {len(processed_items)} files")
        return processed_items
    except Exception as e:
        raise RuntimeError(f"Error loading dataset: {e}")


def normalize_text(text):
    """
    Removes punctuation and converts to lowercase.
    """
    if not isinstance(text, str):
        return ""

    text = re.sub(r"[^\w\s]", "", text)
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text)
    return text.lower().strip()


def calculate_similarity(reference, hypothesis):
    """
    Calculates the Levenshtein Ratio between reference and hypothesis.
    Returns a score between 0 and 100.
    """
    norm_ref = normalize_text(reference)
    norm_hyp = normalize_text(hypothesis)
    score = fuzz.ratio(norm_ref, norm_hyp)
    return score, norm_ref, norm_hyp
