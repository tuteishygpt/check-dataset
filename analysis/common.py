"""Shared helpers for standard and smart analysis modules."""
import numpy as np

import utils
from core.cache import get_cached_dataset, cache_dataset
from core.comparison import select_best_model_result


def has_valid_audio(audio_data) -> bool:
    """Return whether audio_data is usable for transcription."""
    if audio_data is None:
        return False
    if isinstance(audio_data, np.ndarray):
        return audio_data.ndim > 0 and audio_data.size > 0
    if isinstance(audio_data, (list, tuple)):
        return len(audio_data) > 0
    return False


def resolve_audio(ds, result, path_index=None):
    """Resolve audio data for a result row from the dataset.

    Returns (audio_array, sampling_rate) or (None, None).
    """
    item = utils.get_dataset_item(ds, result, path_index=path_index)
    if item is None:
        return None, None
    audio_data, sampling_rate, _ = utils.decode_audio_item(item)
    if not has_valid_audio(audio_data):
        return None, None
    return audio_data, sampling_rate


def init_result_entry(
    idx,
    audio_path,
    ref_text,
    model_name="",
    *,
    audio_data=None,
    sampling_rate=None,
    source_idx=None,
):
    """Build a fresh result record with pending status."""
    entry = {
        "id": idx,
        "source_idx": source_idx if source_idx is not None else idx,
        "path": audio_path,
        "ref_text": ref_text,
        "hyp_text": "",
        "score": 0,
        "norm_ref": "",
        "norm_hyp": "",
        "model_used": model_name,
        "verification_status": "pending",
        "model_results": {},
    }
    if audio_data is not None:
        entry["audio_array"] = audio_data
        entry["sampling_rate"] = sampling_rate
    return entry


def merge_model_result(entry, model_name, hyp_text, score, norm_ref, norm_hyp, similarity_threshold):
    """Record a model transcription and pick the best across all models.

    Mutates *entry* in place.  Returns the (best_model, best_result) tuple
    from select_best_model_result, or (None, None) when nothing qualifies.
    """
    if "model_results" not in entry or not isinstance(entry["model_results"], dict):
        entry["model_results"] = {}

    entry["model_results"][model_name] = {
        "hyp_text": hyp_text,
        "score": score,
        "norm_ref": norm_ref,
        "norm_hyp": norm_hyp,
    }

    best_model, best_result = select_best_model_result(
        entry["model_results"],
        similarity_threshold,
    )
    if best_result:
        entry.update({
            "hyp_text": best_result["hyp_text"],
            "score": best_result["score"],
            "norm_ref": best_result["norm_ref"],
            "norm_hyp": best_result["norm_hyp"],
            "model_used": best_model,
            "verification_status": "correct" if best_result["score"] >= similarity_threshold else "incorrect",
        })
    return best_model, best_result


def load_dataset_cached(dataset_name, limit, hf_token, progress, *, initial_progress=0.0, decode_audio=False):
    """Load a HuggingFace dataset with transparent caching."""
    cached_ds = get_cached_dataset(dataset_name, limit)
    if cached_ds is not None:
        progress(initial_progress, desc=f"Выкарыстоўваю закэшаваны датасет '{dataset_name}'...")
        return cached_ds

    progress(initial_progress, desc=f"Загрузка датасета '{dataset_name}'...")
    ds = utils.load_hf_dataset(dataset_name, limit=limit, hf_token=hf_token, decode_audio=decode_audio)
    cache_dataset(dataset_name, limit, ds)
    progress(min(1.0, initial_progress + 0.05), desc="Датасет закэшаваны")
    return ds
