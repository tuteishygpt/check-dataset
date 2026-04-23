"""Standard analysis with Vertex AI."""
import os
import time
import gradio as gr

import utils
from core.console import safe_print as print
from core.state import (
    get_global_results,
    set_analysis_running,
    set_global_results,
)
from core.cache import get_cached_dataset, cache_dataset
from core.comparison import select_best_model_result
from ui.dashboard import generate_dashboard_outputs
from gemini_api import (
    GeminiIntegrator,
    build_generation_config,
    is_transcription_error,
    supports_batch_inference,
    validate_batch_inference,
    validate_flex_inference,
)
from hf_asr import is_hf_asr_model, get_hf_asr_client, HF_BATCH_SIZE
from analysis.import_export import save_results_csv

ANALYSIS_SCOPE_ALL = "all"
ANALYSIS_SCOPE_PROBLEMATIC = "problematic"
ANALYSIS_SCOPE_PENDING = "pending"
ANALYSIS_SCOPE_PENDING_OR_PROBLEMATIC = "problematic_or_pending"
ANALYSIS_SCOPES = {
    ANALYSIS_SCOPE_ALL,
    ANALYSIS_SCOPE_PROBLEMATIC,
    ANALYSIS_SCOPE_PENDING,
    ANALYSIS_SCOPE_PENDING_OR_PROBLEMATIC,
}


def build_model_generation_config(
    model_name: str,
    temperature: float,
    thinking_budget: int,
    flex_mode: bool,
    location: str,
):
    """Build config for a single-model analysis run."""
    if flex_mode:
        validate_flex_inference(model_name, location=location)

    use_thinking = "thinking" in model_name
    return build_generation_config(
        temperature=temperature,
        thinking_budget=thinking_budget if use_thinking else 0,
        flex_mode=flex_mode,
    )


def normalize_execution_mode(execution_mode: str = None, flex_mode: bool = False) -> str:
    """Normalize execution mode, keeping compatibility with the old flex checkbox."""
    resolved_mode = (execution_mode or ("flex" if flex_mode else "direct")).strip().lower()
    if resolved_mode not in {"direct", "flex", "batch"}:
        raise ValueError(f"Unsupported execution mode: {execution_mode}")
    return resolved_mode


def normalize_analysis_scope(
    analysis_scope: str = None,
    *,
    recheck_problematic: bool = False,
) -> str:
    """Normalize analysis scope while supporting the legacy checkbox flag."""
    if analysis_scope is None:
        return ANALYSIS_SCOPE_PROBLEMATIC if recheck_problematic else ANALYSIS_SCOPE_ALL

    resolved_scope = analysis_scope.strip().lower()
    if resolved_scope not in ANALYSIS_SCOPES:
        raise ValueError(f"Unsupported analysis scope: {analysis_scope}")
    return resolved_scope


def get_analysis_target_indices(
    results,
    analysis_scope: str,
    similarity_threshold: int,
    limit_files: int = 0,
):
    """Return ordered result indices matching the selected scope."""
    scope = normalize_analysis_scope(analysis_scope)

    target_indices = []
    for idx, result in enumerate(results):
        if result is None:
            continue

        is_problematic = (
            result.get("score", 0) < similarity_threshold
            and result.get("verification_status") != "correct"
        )
        is_pending = result.get("verification_status") == "pending"

        if scope == ANALYSIS_SCOPE_ALL:
            target_indices.append(idx)
        elif scope == ANALYSIS_SCOPE_PROBLEMATIC and is_problematic:
            target_indices.append(idx)
        elif scope == ANALYSIS_SCOPE_PENDING and is_pending:
            target_indices.append(idx)
        elif scope == ANALYSIS_SCOPE_PENDING_OR_PROBLEMATIC and (is_problematic or is_pending):
            target_indices.append(idx)

    if limit_files > 0:
        target_indices = target_indices[:limit_files]
    return target_indices


def resolve_execution_mode(model_name: str, execution_mode: str) -> str:
    """Downgrade unsupported execution modes to a safe default."""
    if execution_mode == "batch" and not supports_batch_inference(model_name):
        print(
            "Batch mode is unsupported for model "
            f"{model_name}; falling back to direct mode."
        )
        return "direct"
    return execution_mode


def is_gemini_model(model_name: str) -> bool:
    """Return whether the recognition model should run through Vertex Gemini."""
    return bool(model_name) and model_name.startswith("gemini-")


def _load_analysis_dataset(dataset_name: str, limit: int, hf_token: str, progress, *, initial_progress: float = 0.0):
    """Load analysis dataset with cache reuse."""
    cached_ds = get_cached_dataset(dataset_name, limit)
    if cached_ds is not None:
        progress(initial_progress, desc=f"Ð’Ñ‹ÐºÐ°Ñ€Ñ‹ÑÑ‚Ð¾ÑžÐ²Ð°ÑŽ Ð·Ð°ÐºÑÑˆÐ°Ð²Ð°Ð½Ñ‹ Ð´Ð°Ñ‚Ð°ÑÐµÑ‚ '{dataset_name}'...")
        return cached_ds

    progress(initial_progress, desc=f"Ð—Ð°Ð³Ñ€ÑƒÐ·ÐºÐ° Ð´Ð°Ñ‚Ð°ÑÐµÑ‚Ð° '{dataset_name}'...")
    ds = utils.load_hf_dataset(dataset_name, limit=limit, hf_token=hf_token, decode_audio=False)
    cache_dataset(dataset_name, limit, ds)
    progress(min(1.0, initial_progress + 0.05), desc="Ð”Ð°Ñ‚Ð°ÑÐµÑ‚ Ð·Ð°ÐºÑÑˆÐ°Ð²Ð°Ð½Ñ‹")
    return ds


def _get_result_audio(ds, result, path_index=None):
    """Resolve audio data for a result row without storing it in global state."""
    item = utils.get_dataset_item(ds, result, path_index=path_index)
    if item is None:
        return None, None

    audio_data, sampling_rate, _ = utils.decode_audio_item(item)
    if audio_data is None or (hasattr(audio_data, "__len__") and len(audio_data) == 0):
        return None, None
    return audio_data, sampling_rate


def _has_audio_data(audio_data) -> bool:
    """Return whether a cached result contains usable audio data."""
    if audio_data is None:
        return False
    if hasattr(audio_data, "__len__") and len(audio_data) == 0:
        return False
    return True


def run_analysis(
    dataset_name: str,
    model_name: str,
    limit_files: int,
    temperature: float,
    thinking_budget: int,
    similarity_threshold: int,
    execution_mode: str = None,
    analysis_scope: str = None,
    recheck_problematic: bool = False,
    hf_token: str = None,
    progress=gr.Progress(),
    flex_mode: bool = False,
):
    from core.state import set_stop_requested

    set_stop_requested(False)
    set_analysis_running(True)

    analysis_scope = normalize_analysis_scope(
        analysis_scope,
        recheck_problematic=recheck_problematic,
    )
    limit_files = int(float(limit_files)) if limit_files else 0
    thinking_budget = int(float(thinking_budget)) if thinking_budget else 0
    similarity_threshold = int(float(similarity_threshold)) if similarity_threshold else 90
    temperature = float(temperature)

    try:
        global_results = get_global_results()

        if is_hf_asr_model(model_name):
            outputs = _run_hf_asr_analysis(
                model_name,
                dataset_name,
                limit_files,
                analysis_scope,
                similarity_threshold,
                hf_token,
                progress,
            )
            save_results_csv(dataset_name, auto_prefix=True)
            return outputs

        if not is_gemini_model(model_name):
            raise gr.Error(
                "Unsupported recognition model. Choose a Gemini model or SeamlessM4T-v2 (HF)."
            )

        execution_mode = normalize_execution_mode(execution_mode, flex_mode=flex_mode)
        execution_mode = resolve_execution_mode(model_name, execution_mode)

        gemini_tool = GeminiIntegrator()
        gen_config = build_model_generation_config(
            model_name=model_name,
            temperature=temperature,
            thinking_budget=thinking_budget,
            flex_mode=execution_mode == "flex",
            location=gemini_tool.location,
        )

        if execution_mode == "batch":
            validate_batch_inference(model_name)
            outputs = _run_vertex_batch_analysis(
                gemini_tool,
                model_name,
                dataset_name,
                limit_files,
                similarity_threshold,
                gen_config,
                hf_token,
                progress,
                analysis_scope=analysis_scope,
            )
        elif analysis_scope == ANALYSIS_SCOPE_ALL or (
            analysis_scope in {ANALYSIS_SCOPE_PENDING, ANALYSIS_SCOPE_PENDING_OR_PROBLEMATIC}
            and not global_results
        ):
            outputs = _run_fresh_analysis(
                gemini_tool,
                model_name,
                dataset_name,
                limit_files,
                analysis_scope,
                similarity_threshold,
                gen_config,
                hf_token,
                progress,
            )
        else:
            outputs = _run_recheck_analysis(
                gemini_tool,
                model_name,
                dataset_name,
                limit_files,
                analysis_scope,
                similarity_threshold,
                gen_config,
                hf_token,
                progress,
            )
        return outputs
    except Exception as e:
        print(f"Analysis failed: {e}")
        save_results_csv(dataset_name, auto_prefix=True)
        raise gr.Error(f"Error: {e}")
    finally:
        set_analysis_running(False)
        save_results_csv(dataset_name, auto_prefix=True)


def _run_vertex_batch_analysis(
    gemini_tool,
    model_name,
    dataset_name,
    limit_files,
    similarity_threshold,
    gen_config,
    hf_token,
    progress,
    analysis_scope=ANALYSIS_SCOPE_ALL,
):
    """Run fresh analysis through Vertex Batch API with GCS staging."""
    global_results = get_global_results()
    target_indices = []
    if global_results:
        if analysis_scope == ANALYSIS_SCOPE_ALL:
            target_indices = [
                i for i, result in enumerate(global_results)
                if result is not None and result.get("verification_status") == "pending"
            ]
        else:
            target_indices = get_analysis_target_indices(
                global_results,
                analysis_scope,
                similarity_threshold,
                limit_files=limit_files,
            )

    if target_indices:
        limit = None
        ds = _load_analysis_dataset(dataset_name, limit, hf_token, progress, initial_progress=0.0)
        path_index = utils.build_dataset_path_index(ds)

        results = global_results
        batch_records = []
        for idx in target_indices:
            result = results[idx]
            audio_data = result.get("audio_array")
            sampling_rate = result.get("sampling_rate")
            if not _has_audio_data(audio_data):
                audio_data, sampling_rate = _get_result_audio(ds, result, path_index=path_index)
                if not _has_audio_data(audio_data):
                    continue
                results[idx]["audio_array"] = audio_data
                results[idx]["sampling_rate"] = sampling_rate

            batch_records.append(
                {
                    "id": idx,
                    "path": result.get("path", ""),
                    "ref_text": result.get("ref_text", ""),
                    "audio_array": audio_data,
                    "sampling_rate": sampling_rate,
                    "existing_result": True,
                }
            )
    elif global_results and analysis_scope != ANALYSIS_SCOPE_ALL:
        results = global_results
        batch_records = []
    else:
        limit = int(limit_files) if limit_files > 0 else None
        ds = _load_analysis_dataset(dataset_name, limit, hf_token, progress, initial_progress=0.0)
        results = []
        batch_records = []

        for idx, item in enumerate(ds):
            ref_text = item.get("sentence") or item.get("text") or item.get("transcription") or item.get("transcript") or ""
            audio_data, sampling_rate, audio_path = utils.decode_audio_item(item)
            result_record = {
                "id": idx,
                "path": audio_path,
                "ref_text": ref_text,
                "hyp_text": "",
                "score": 0,
                "norm_ref": "",
                "norm_hyp": "",
                "audio_array": audio_data,
                "sampling_rate": sampling_rate,
                "model_used": model_name,
                "verification_status": "pending",
                "model_results": {},
            }
            results.append(result_record)
            batch_records.append(
                {
                    "id": idx,
                    "path": result_record["path"],
                    "ref_text": ref_text,
                    "audio_array": result_record["audio_array"],
                    "sampling_rate": result_record["sampling_rate"],
                }
            )

    if not batch_records:
        return generate_dashboard_outputs(similarity_threshold)

    set_global_results(results)
    progress(0.1, desc=f"Vertex Batch: upload and submit for {len(batch_records)} files...")
    batch_outputs = gemini_tool.transcribe_audio_batch(
        model_name,
        batch_records,
        config=gen_config,
    )

    total_records = max(1, len(batch_records))
    for position, record in enumerate(batch_records):
        from core.state import get_stop_requested

        if get_stop_requested():
            print("ðŸ›‘ ÐÐ½Ð°Ð»Ñ–Ð· ÑÐ¿Ñ‹Ð½ÐµÐ½Ñ‹ ÐºÐ°Ñ€Ñ‹ÑÑ‚Ð°Ð»ÑŒÐ½Ñ–ÐºÐ°Ð¼")
            break

        idx = int(record["id"])
        hyp_text = batch_outputs.get(idx, "Error: Missing batch output")
        progress(0.1 + (position + 1) / total_records * 0.9, desc=f"Vertex Batch: result {position + 1}/{len(batch_records)}")

        if is_transcription_error(hyp_text):
            if record.get("existing_result"):
                continue
            results[idx].update({
                "hyp_text": "",
                "score": 0,
                "norm_ref": "",
                "norm_hyp": "",
                "model_used": model_name,
                "verification_status": "pending",
            })
            continue

        score, norm_ref, norm_hyp = utils.calculate_similarity(record["ref_text"], hyp_text)
        model_result = {
            "hyp_text": hyp_text,
            "score": score,
            "norm_ref": norm_ref,
            "norm_hyp": norm_hyp,
        }
        if record.get("existing_result"):
            if "model_results" not in results[idx] or not isinstance(results[idx]["model_results"], dict):
                results[idx]["model_results"] = {}
            results[idx]["model_results"][model_name] = model_result
            best_model, best_result = select_best_model_result(
                results[idx]["model_results"],
                similarity_threshold,
            )
            if not best_result:
                continue
            results[idx].update({
                "hyp_text": best_result["hyp_text"],
                "score": best_result["score"],
                "norm_ref": best_result["norm_ref"],
                "norm_hyp": best_result["norm_hyp"],
                "model_used": best_model,
                "verification_status": "correct" if best_result["score"] >= similarity_threshold else "incorrect",
            })
            continue

        results[idx].update({
            "hyp_text": hyp_text,
            "score": score,
            "norm_ref": norm_ref,
            "norm_hyp": norm_hyp,
            "model_used": model_name,
            "verification_status": "correct" if score >= similarity_threshold else "incorrect",
            "model_results": {
                model_name: model_result
            },
        })

        if (position + 1) % 20 == 0:
            set_global_results(results)
            save_results_csv(dataset_name, auto_prefix=False)

    set_global_results(results)
    return generate_dashboard_outputs(similarity_threshold)


def _run_recheck_analysis(
    gemini_tool, model_name, dataset_name, limit_files,
    analysis_scope=ANALYSIS_SCOPE_PROBLEMATIC, similarity_threshold=None, gen_config=None,
    hf_token=None, progress=None
):
    """Run analysis for a selected subset of existing results."""
    global_results = get_global_results()
    
    if not global_results:
        gr.Warning("Няма вынікаў для пераправеркі.")
        return generate_dashboard_outputs(similarity_threshold)
    
    target_indices = get_analysis_target_indices(
        global_results,
        analysis_scope,
        similarity_threshold,
        limit_files=limit_files,
    )
    
    if not target_indices:
        gr.Info("Няма праблемных файлаў для пераправеркі.")
        return generate_dashboard_outputs(similarity_threshold)

    # Load dataset to get audio for files that might be missing it
    limit = None
    cached_ds = get_cached_dataset(dataset_name, limit)
    if cached_ds is not None:
        progress(0, desc=f"Выкарыстоўваю закэшаваны датасет '{dataset_name}'...")
        ds = cached_ds
    else:
        progress(0, desc=f"Загрузка датасета '{dataset_name}'...")
        ds = utils.load_hf_dataset(dataset_name, limit=limit, hf_token=hf_token)
        cache_dataset(dataset_name, limit, ds)
        progress(0.05, desc=f"Датасет закэшаваны")
    
    path_index = utils.build_dataset_path_index(ds)

    progress(0.1, desc=f"Пераправерка {len(target_indices)} файлаў...")
    
    for j, idx in enumerate(target_indices):
        from core.state import get_stop_requested
        if get_stop_requested():
            print("🛑 Аналіз спынены карыстальнікам")
            break
        progress(0.1 + (j + 1) / len(target_indices) * 0.9, desc=f"Праверка {j+1}/{len(target_indices)}")
        
        result = global_results[idx]
        audio_data = result.get('audio_array')
        sampling_rate = result.get('sampling_rate')
        ref_text = result.get('ref_text', "")
        
        # If audio is missing, try to fetch from dataset
        if not _has_audio_data(audio_data):
            audio_data, sampling_rate = _get_result_audio(ds, result, path_index=path_index)
            if _has_audio_data(audio_data):
                global_results[idx]['audio_array'] = audio_data
                global_results[idx]['sampling_rate'] = sampling_rate
            else:
                print(f"Problematic Recheck: Skipping index {idx}, path '{result.get('path', '')}', id {result.get('id')}: Audio not found in dataset.")
                continue

        hyp_text = gemini_tool.transcribe_audio(model_name, audio_data, sampling_rate, config=gen_config)
        if is_transcription_error(hyp_text):
            print(f"⚠️ Прапускаем (памылка распазнавання): {result.get('path')} | {hyp_text}")
            continue
        score, norm_ref, norm_hyp = utils.calculate_similarity(ref_text, hyp_text)

        print(f"🔄 Updated: {result.get('path')} | Score: {result.get('score')} -> {score} | Text: {hyp_text}")

        # Save model result
        if 'model_results' not in global_results[idx]:
            global_results[idx]['model_results'] = {}
        
        global_results[idx]['model_results'][model_name] = {
            "hyp_text": hyp_text,
            "score": score,
            "norm_ref": norm_ref,
            "norm_hyp": norm_hyp
        }
        
        # Select best result from all models
        best_model, best_result = select_best_model_result(
            global_results[idx]['model_results'], 
            similarity_threshold
        )
        
        if best_result:
            global_results[idx].update({
                "hyp_text": best_result['hyp_text'],
                "score": best_result['score'],
                "norm_ref": best_result['norm_ref'],
                "norm_hyp": best_result['norm_hyp'],
                "model_used": best_model,
                "verification_status": "correct" if best_result['score'] >= similarity_threshold else "incorrect"
            })
        
        # Periodic save every 20 items
        if (j + 1) % 20 == 0:
            save_results_csv(dataset_name, auto_prefix=False)

    return generate_dashboard_outputs(similarity_threshold)


def _run_fresh_analysis(
    gemini_tool, model_name, dataset_name, limit_files,
    analysis_scope=ANALYSIS_SCOPE_ALL, similarity_threshold=None, gen_config=None,
    hf_token=None, progress=None
):
    """Run fresh analysis, resuming from pending files if they exist."""
    global_results = get_global_results()

    # Check for pending (unprocessed) files from a previous interrupted run
    pending_indices = [
        i for i, r in enumerate(global_results)
        if r is not None and r.get('verification_status') == 'pending'
    ]

    if analysis_scope == ANALYSIS_SCOPE_PENDING and pending_indices:
        # Resume: process only pending files
        print(f"▶️ Аднаўленне аналізу: знойдзена {len(pending_indices)} неапрацаваных файлаў (pending)")
        if limit_files > 0:
            pending_indices = pending_indices[:limit_files]

        # Load dataset to get audio for pending items
        limit = None
        cached_ds = get_cached_dataset(dataset_name, limit)
        if cached_ds is not None:
            progress(0, desc=f"Выкарыстоўваю закэшаваны датасет '{dataset_name}'...")
            ds = cached_ds
        else:
            progress(0, desc=f"Загрузка датасета '{dataset_name}'...")
            ds = utils.load_hf_dataset(dataset_name, limit=limit, hf_token=hf_token)
            cache_dataset(dataset_name, limit, ds)
            progress(0.1, desc="Dataset cached for reuse")

        path_index = utils.build_dataset_path_index(ds)

        results = global_results
        progress(0.1, desc=f"Апрацоўка {len(pending_indices)} неапрацаваных файлаў...")

        for j, idx in enumerate(pending_indices):
            from core.state import get_stop_requested
            if get_stop_requested():
                print("🛑 Аналіз спынены карыстальнікам")
                break
            progress(0.1 + (j + 1) / len(pending_indices) * 0.9, desc=f"Апрацоўка {j+1}/{len(pending_indices)}")

            result = results[idx]
            audio_data = result.get('audio_array')
            sampling_rate = result.get('sampling_rate')
            ref_text = result.get('ref_text', "")

            # Fetch audio from dataset if missing
            if not _has_audio_data(audio_data):
                audio_data, sampling_rate = _get_result_audio(ds, result, path_index=path_index)
                if _has_audio_data(audio_data):
                    results[idx]['audio_array'] = audio_data
                    results[idx]['sampling_rate'] = sampling_rate
                else:
                    print(f"Fresh Resume: Skipping index {idx}, audio not found.")
                    continue

            hyp_text = gemini_tool.transcribe_audio(model_name, audio_data, sampling_rate, config=gen_config)
            if is_transcription_error(hyp_text):
                print(f"⚠️ Прапускаем (памылка распазнавання): {result.get('path', f'idx={idx}')} | {hyp_text}")
                continue
            score, norm_ref, norm_hyp = utils.calculate_similarity(ref_text, hyp_text)

            if 'model_results' not in results[idx]:
                results[idx]['model_results'] = {}
            results[idx]['model_results'][model_name] = {
                "hyp_text": hyp_text, "score": score,
                "norm_ref": norm_ref, "norm_hyp": norm_hyp
            }
            results[idx].update({
                "hyp_text": hyp_text, "score": score,
                "norm_ref": norm_ref, "norm_hyp": norm_hyp,
                "audio_array": audio_data, "sampling_rate": sampling_rate,
                "model_used": model_name,
                "verification_status": "correct" if score >= similarity_threshold else "incorrect"
            })

            if (j + 1) % 20 == 0:
                set_global_results(results)
                save_results_csv(dataset_name, auto_prefix=False)

        set_global_results(results)
        return generate_dashboard_outputs(similarity_threshold)

    # No pending files — full fresh run
    limit = int(limit_files) if limit_files > 0 else None

    cached_ds = get_cached_dataset(dataset_name, limit)
    if cached_ds is not None:
        progress(0, desc=f"Выкарыстоўваю закэшаваны датасет '{dataset_name}'...")
        ds = cached_ds
    else:
        progress(0, desc=f"Загрузка датасета '{dataset_name}'...")
        ds = utils.load_hf_dataset(dataset_name, limit=limit, hf_token=hf_token, decode_audio=False)
        cache_dataset(dataset_name, limit, ds)
        progress(0.1, desc=f"Датасет закэшаваны для паўторнага выкарыстання")

    results = []

    for idx, item in enumerate(ds):
        from core.state import get_stop_requested
        if get_stop_requested():
            print("🛑 Аналіз спынены карыстальнікам")
            break
        progress((idx + 1) / len(ds), desc=f"Апрацоўка файла {idx+1}/{len(ds)}")

        audio_data, sampling_rate, audio_path = utils.decode_audio_item(item)
        ref_text = utils.get_record_text(item)

        hyp_text = gemini_tool.transcribe_audio(model_name, audio_data, sampling_rate, config=gen_config)
        if is_transcription_error(hyp_text):
            print(f"⚠️ Прапускаем (памылка распазнавання): {item['audio']['path']} | {hyp_text}")
            # Add a pending record so it can be retried later
            results.append({
                "id": idx,
                "source_idx": idx,
                "path": audio_path,
                "ref_text": ref_text,
                "hyp_text": "",
                "score": 0,
                "norm_ref": "",
                "norm_hyp": "",
                "model_used": model_name,
                "verification_status": "pending",
                "model_results": {}
            })
            continue
        score, norm_ref, norm_hyp = utils.calculate_similarity(ref_text, hyp_text)

        results.append({
            "id": idx,
            "source_idx": idx,
            "path": audio_path,
            "ref_text": ref_text,
            "hyp_text": hyp_text,
            "score": score,
            "norm_ref": norm_ref,
            "norm_hyp": norm_hyp,
            "model_used": model_name,
            "verification_status": "correct" if score >= similarity_threshold else "incorrect",
            "model_results": {
                model_name: {
                    "hyp_text": hyp_text,
                    "score": score,
                    "norm_ref": norm_ref,
                    "norm_hyp": norm_hyp
                }
            }
        })

        # Periodic save every 20 items
        if (idx + 1) % 20 == 0:
            set_global_results(results)
            save_results_csv(dataset_name, auto_prefix=False)

    set_global_results(results)
    return generate_dashboard_outputs(similarity_threshold)


def _run_hf_asr_analysis(
    model_name: str,
    dataset_name: str,
    limit_files: int,
    analysis_scope: str,
    similarity_threshold: int,
    hf_token: str,
    progress
):
    """Run analysis using Hugging Face ASR model."""
    global_results = get_global_results()
    
    try:
        hf_client = get_hf_asr_client(model_name, hf_token=hf_token)
        progress(0.05, desc=f"Падключэнне да HF Space: {model_name}...")
    except Exception as e:
        raise gr.Error(f"Памылка падключэння да HF: {e}")
    
    if analysis_scope == ANALYSIS_SCOPE_ALL or (
        analysis_scope in {ANALYSIS_SCOPE_PENDING, ANALYSIS_SCOPE_PENDING_OR_PROBLEMATIC}
        and not global_results
    ):
        return _run_hf_fresh_analysis(
            hf_client, model_name, dataset_name, limit_files,
            analysis_scope, similarity_threshold, hf_token, progress
        )
    else:
        return _run_hf_recheck_analysis(
            hf_client, model_name, dataset_name, limit_files,
            analysis_scope, similarity_threshold, hf_token, progress
        )


def _run_hf_fresh_analysis(
    hf_client, model_name, dataset_name, limit_files,
    analysis_scope, similarity_threshold, hf_token, progress
):
    """Run fresh HF ASR analysis, resuming from pending files if they exist."""
    global_results = get_global_results()

    # Check for pending (unprocessed) files from a previous interrupted run
    pending_indices = [
        i for i, r in enumerate(global_results)
        if r is not None and r.get('verification_status') == 'pending'
    ]

    if analysis_scope == ANALYSIS_SCOPE_PENDING and pending_indices:
        # Resume mode: process only pending items
        print(f"▶️ HF ASR: аднаўленне — знойдзена {len(pending_indices)} неапрацаваных файлаў (pending)")
        if limit_files > 0:
            pending_indices = pending_indices[:limit_files]

        # Load dataset to fetch missing audio
        limit = None
        cached_ds = get_cached_dataset(dataset_name, limit)
        if cached_ds is not None:
            progress(0, desc=f"Выкарыстоўваю закэшаваны датасет '{dataset_name}'...")
            ds = cached_ds
        else:
            progress(0, desc=f"Загрузка датасета '{dataset_name}'...")
            ds = utils.load_hf_dataset(dataset_name, limit=limit, hf_token=hf_token, decode_audio=False)
            cache_dataset(dataset_name, limit, ds)
            progress(0.1, desc="Dataset cached for reuse")

        path_index = utils.build_dataset_path_index(ds)

        # Build audio map
        audio_map = {}
        for item in ds:
            path = item['audio']['path']
            if path:
                audio_map[os.path.basename(path)] = item
                audio_map[path] = item

        # Collect audio data for pending items
        items_to_process = []
        results = global_results
        for idx in pending_indices:
            result = results[idx]
            item = utils.get_dataset_item(ds, result, path_index=path_index)
            if item is None:
                print(f"HF Fresh Resume: Skipping index {idx}, audio not found.")
                continue

            items_to_process.append({
                "idx": idx,
                "source_idx": int(result.get("source_idx", idx)),
                "path": result.get('path', ''),
                "ref_text": result.get('ref_text', ""),
            })

        total_items = len(items_to_process)
        batch_size = HF_BATCH_SIZE
        num_batches = (total_items + batch_size - 1) // batch_size

        progress(0.1, desc=f"Апрацоўка {total_items} файлаў (HF ASR, {num_batches} пакетаў)...")

        for batch_num in range(num_batches):
            from core.state import get_stop_requested
            if get_stop_requested():
                print("🛑 Аналіз спынены карыстальнікам")
                break
            if batch_num > 0:
                print(f"⏳ Чакаем 5с перад наступным пакетам...")
                time.sleep(5)

            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, total_items)
            batch_items = items_to_process[start_idx:end_idx]

            progress_val = 0.1 + (batch_num / num_batches) * 0.9
            progress(progress_val, desc=f"Пакет {batch_num + 1}/{num_batches}: {len(batch_items)} файлаў (HF ASR)...")

            batch_audio = []
            for item in batch_items:
                ds_item = ds[item["source_idx"]]
                audio_data, sampling_rate, _ = utils.decode_audio_item(ds_item)
                batch_audio.append((item["idx"], audio_data, sampling_rate))

            try:
                transcriptions = hf_client.transcribe_batch(batch_audio)
            except RuntimeError as e:
                if str(e) == "QUOTA_EXCEEDED":
                    print("🛑 Спыненне: перавышанне квоты Pro GPU. Захаванне...")
                    break
                transcriptions = {}

            transcribed_count = 0
            for item in batch_items:
                idx = item["idx"]
                ref_text = item["ref_text"]
                hyp_text = transcriptions.get(idx, "")
                if not hyp_text or is_transcription_error(hyp_text):
                    if is_transcription_error(hyp_text):
                        print(f"⚠️ HF ASR: прапускаем памылку для idx={idx}: {hyp_text}")
                    continue
                transcribed_count += 1
                score, norm_ref, norm_hyp = utils.calculate_similarity(ref_text, hyp_text)
                results[idx].update({
                    "hyp_text": hyp_text, "score": score,
                    "norm_ref": norm_ref, "norm_hyp": norm_hyp,
                    "model_used": model_name,
                    "verification_status": "correct" if score >= similarity_threshold else "incorrect",
                    "model_results": {model_name: {"hyp_text": hyp_text, "score": score,
                                                   "norm_ref": norm_ref, "norm_hyp": norm_hyp}}
                })

            print(f"✅ Пакет {batch_num + 1}/{num_batches}: {transcribed_count}/{len(batch_items)} транскрыбавана")
            set_global_results(results)
            save_results_csv(dataset_name, auto_prefix=False)

        set_global_results(results)
        return generate_dashboard_outputs(similarity_threshold)

    # No pending files — full fresh run
    limit = int(limit_files) if limit_files > 0 else None

    cached_ds = get_cached_dataset(dataset_name, limit)
    if cached_ds is not None:
        progress(0, desc=f"Выкарыстоўваю закэшаваны датасет '{dataset_name}'...")
        ds = cached_ds
    else:
        progress(0, desc=f"Загрузка датасета '{dataset_name}'...")
        ds = utils.load_hf_dataset(dataset_name, limit=limit, hf_token=hf_token, decode_audio=False)
        cache_dataset(dataset_name, limit, ds)
        progress(0.1, desc=f"Датасет закэшаваны для паўторнага выкарыстання")

    # Pre-collect lightweight metadata only. Audio is decoded per batch.
    all_items = []
    for idx, item in enumerate(ds):
        ref_text = utils.get_record_text(item)
        all_items.append({
            "idx": idx,
            "source_idx": idx,
            "path": utils.get_audio_path(item),
            "ref_text": ref_text
        })

    total_items = len(all_items)
    # Pre-allocate with pending status
    results = []
    for item in all_items:
        results.append({
            "id": item["idx"],
            "source_idx": item["source_idx"],
            "path": item["path"],
            "ref_text": item["ref_text"],
            "hyp_text": "",
            "score": 0,
            "norm_ref": "",
            "norm_hyp": "",
            "model_used": model_name,
            "verification_status": "pending",
            "model_results": {}
        })
    set_global_results(results)

    # Process in batches of HF_BATCH_SIZE (100)
    batch_size = HF_BATCH_SIZE
    num_batches = (total_items + batch_size - 1) // batch_size

    for batch_num in range(num_batches):
        from core.state import get_stop_requested
        if get_stop_requested():
            print("🛑 Аналіз спынены карыстальнікам")
            break
        # Delay between batches to avoid rate limiting (skip for first batch)
        if batch_num > 0:
            print(f"⏳ Чакаем 5с перад наступным пакетам...")
            time.sleep(5)

        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, total_items)
        batch_items = all_items[start_idx:end_idx]

        progress_val = 0.1 + (batch_num / num_batches) * 0.9
        progress(progress_val, desc=f"Пакет {batch_num + 1}/{num_batches}: апрацоўка {len(batch_items)} файлаў (HF ASR)...")

        # Prepare batch for transcription: (key, audio_array, sampling_rate)
        batch_audio = []
        for item in batch_items:
            ds_item = ds[item["source_idx"]]
            audio_data, sampling_rate, _ = utils.decode_audio_item(ds_item)
            batch_audio.append((item["idx"], audio_data, sampling_rate))

        # Send batch to HF ASR (retry logic is inside transcribe_batch)
        try:
            transcriptions = hf_client.transcribe_batch(batch_audio)
        except RuntimeError as e:
            if str(e) == "QUOTA_EXCEEDED":
                print("🛑 Спыненне: перавышанне квоты Pro GPU. Захаванне ўжо апрацаваных даных...")
                break
            transcriptions = {}

        # Process results - only save successful transcriptions
        transcribed_count = 0
        for item in batch_items:
            idx = item["idx"]
            ref_text = item["ref_text"]
            hyp_text = transcriptions.get(idx, "")

            # Only record result if transcription was successful (not empty/error)
            if hyp_text and not is_transcription_error(hyp_text):
                score, norm_ref, norm_hyp = utils.calculate_similarity(ref_text, hyp_text)
                transcribed_count += 1

                results[idx] = {
                    "id": idx,
                    "source_idx": item["source_idx"],
                    "path": item["path"],
                    "ref_text": ref_text,
                    "hyp_text": hyp_text,
                    "score": score,
                    "norm_ref": norm_ref,
                    "norm_hyp": norm_hyp,
                    "model_used": model_name,
                    "verification_status": "correct" if score >= similarity_threshold else "incorrect",
                    "model_results": {
                        model_name: {
                            "hyp_text": hyp_text,
                            "score": score,
                            "norm_ref": norm_ref,
                            "norm_hyp": norm_hyp
                        }
                    }
                }
            # Skip items with no transcription result

        print(f"✅ Пакет {batch_num + 1}/{num_batches} завершаны: {transcribed_count}/{len(batch_items)} транскрыбавана")
        set_global_results(results)
        save_results_csv(dataset_name, auto_prefix=False)

    set_global_results(results)
    return generate_dashboard_outputs(similarity_threshold)


def _run_hf_recheck_analysis(
    hf_client, model_name, dataset_name, limit_files,
    analysis_scope, similarity_threshold, hf_token, progress
):
    """Run recheck of problematic files using HF ASR with batch processing."""
    global_results = get_global_results()
    
    if not global_results:
        gr.Warning("Няма вынікаў для пераправеркі.")
        return generate_dashboard_outputs(similarity_threshold)
    
    target_indices = get_analysis_target_indices(
        global_results,
        analysis_scope,
        similarity_threshold,
        limit_files=limit_files,
    )
    
    if not target_indices:
        gr.Info("Няма праблемных файлаў для пераправеркі.")
        return generate_dashboard_outputs(similarity_threshold)
    
    # Load dataset to get audio for files that might be missing it
    limit = None
    cached_ds = get_cached_dataset(dataset_name, limit)
    if cached_ds is not None:
        progress(0, desc=f"Выкарыстоўваю закэшаваны датасет '{dataset_name}'...")
        ds = cached_ds
    else:
        progress(0, desc=f"Загрузка датасета '{dataset_name}'...")
        ds = utils.load_hf_dataset(dataset_name, limit=limit, hf_token=hf_token)
        cache_dataset(dataset_name, limit, ds)
        progress(0.05, desc=f"Датасет закэшаваны")
    
    path_index = utils.build_dataset_path_index(ds)
    
    # Collect all items to process with their audio data
    items_to_process = []
    for idx in target_indices:
        result = global_results[idx]
        audio_data = result.get('audio_array')
        sampling_rate = result.get('sampling_rate')
        ref_text = result.get('ref_text', "")
        
        # If audio is missing, try to fetch from dataset
        if not _has_audio_data(audio_data):
            audio_data, sampling_rate = _get_result_audio(ds, result, path_index=path_index)
            if _has_audio_data(audio_data):
                global_results[idx]['audio_array'] = audio_data
                global_results[idx]['sampling_rate'] = sampling_rate
            else:
                print(f"HF Recheck: Skipping index {idx}, path '{result.get('path', '')}': Audio not found.")
                continue
        
        items_to_process.append({
            "idx": idx,
            "audio_data": audio_data,
            "sampling_rate": sampling_rate,
            "ref_text": ref_text
        })
    
    if not items_to_process:
        gr.Info("Няма файлаў з аўдыя для пераправеркі.")
        return generate_dashboard_outputs(similarity_threshold)
    
    # Process in batches of HF_BATCH_SIZE (100)
    batch_size = HF_BATCH_SIZE
    total_items = len(items_to_process)
    num_batches = (total_items + batch_size - 1) // batch_size
    
    progress(0.1, desc=f"Пераправерка {total_items} файлаў у {num_batches} пакетах (HF ASR)...")
    
    for batch_num in range(num_batches):
        from core.state import get_stop_requested
        if get_stop_requested():
            print("🛑 Аналіз спынены карыстальнікам")
            break
        # Delay between batches to avoid rate limiting (skip for first batch)
        if batch_num > 0:
            print(f"⏳ Чакаем 5с перад наступным пакетам...")
            time.sleep(5)
        
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, total_items)
        batch_items = items_to_process[start_idx:end_idx]
        
        progress_val = 0.1 + (batch_num / num_batches) * 0.9
        progress(progress_val, desc=f"Пакет {batch_num + 1}/{num_batches}: апрацоўка {len(batch_items)} файлаў...")
        
        # Prepare batch for transcription: (key, audio_array, sampling_rate)
        batch_audio = [
            (item["idx"], item["audio_data"], item["sampling_rate"])
            for item in batch_items
        ]
        
        # Send batch to HF ASR (retry logic is inside transcribe_batch)
        try:
            transcriptions = hf_client.transcribe_batch(batch_audio)
        except RuntimeError as e:
            if str(e) == "QUOTA_EXCEEDED":
                print("🛑 Спыненне: перавышанне квоты Pro GPU. Захаванне ўжо апрацаваных даных...")
                break
            transcriptions = {}
        
        # Process results - only save successful transcriptions
        transcribed_count = 0
        for item in batch_items:
            idx = item["idx"]
            ref_text = item["ref_text"]
            hyp_text = transcriptions.get(idx, "")
            
            if not hyp_text or is_transcription_error(hyp_text):
                if is_transcription_error(hyp_text):
                    print(f"⚠️ HF Recheck: прапускаем памылку для idx={idx}: {hyp_text}")
                continue
            
            transcribed_count += 1
            score, norm_ref, norm_hyp = utils.calculate_similarity(ref_text, hyp_text)
            
            print(f"🔄 HF Updated: {global_results[idx].get('path')} | Score: {global_results[idx].get('score')} -> {score}")
            
            # Save model result
            if 'model_results' not in global_results[idx]:
                global_results[idx]['model_results'] = {}
            
            global_results[idx]['model_results'][model_name] = {
                "hyp_text": hyp_text,
                "score": score,
                "norm_ref": norm_ref,
                "norm_hyp": norm_hyp
            }
            
            # Select best result from all models
            best_model, best_result = select_best_model_result(
                global_results[idx]['model_results'], 
                similarity_threshold
            )
            
            if best_result:
                global_results[idx].update({
                    "hyp_text": best_result['hyp_text'],
                    "score": best_result['score'],
                    "norm_ref": best_result['norm_ref'],
                    "norm_hyp": best_result['norm_hyp'],
                    "model_used": best_model,
                    "verification_status": "correct" if best_result['score'] >= similarity_threshold else "incorrect"
                })
        
        print(f"✅ Пакет {batch_num + 1}/{num_batches} завершаны: {transcribed_count}/{len(batch_items)} транскрыбавана")
        save_results_csv(dataset_name, auto_prefix=False)
    
    return generate_dashboard_outputs(similarity_threshold)


