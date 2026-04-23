"""Smart analysis - multi-model iterative processing."""
import os
import gradio as gr

import utils
from core.console import safe_print as print
from core.state import (
    append_analysis_log,
    get_global_results,
    set_analysis_running,
    set_global_results,
)
from core.cache import get_cached_dataset, cache_dataset
from core.comparison import select_best_model_result
from ui.dashboard import generate_dashboard_outputs
from gemini_api import GeminiIntegrator, build_generation_config, validate_flex_inference
from analysis.import_export import save_results_csv
from gemini_api import is_transcription_error
from analysis.standard import (
    ANALYSIS_SCOPE_ALL,
    ANALYSIS_SCOPE_PENDING,
    ANALYSIS_SCOPE_PENDING_OR_PROBLEMATIC,
    ANALYSIS_SCOPE_PROBLEMATIC,
    get_analysis_target_indices,
    normalize_analysis_scope,
)

STANDARD_SMART_MODELS = [
    ("gemini-2.5-flash-lite", "Step 1/4: Flash-Lite"),
    ("gemini-3.1-flash-lite-preview", "Step 2/4: Flash-Lite 3.1"),
    ("gemini-2.5-flash", "Step 3/4: Flash"),
    ("gemini-2.5-pro", "Step 4/4: Gemini-2.5-Pro"),
]
FLEX_SMART_MODELS = [
    ("gemini-3.1-flash-lite-preview", "Step 1/3: Gemini-3.1 Flash-Lite Preview"),
    ("gemini-3-flash-preview", "Step 2/3: Gemini-3 Flash Preview"),
    ("gemini-3.1-pro-preview", "Step 3/3: Gemini-3.1 Pro Preview"),
]


def get_smart_models(flex_mode: bool):
    """Return the model chain for smart analysis."""
    return FLEX_SMART_MODELS if flex_mode else STANDARD_SMART_MODELS


def build_smart_generation_config(temperature: float, flex_mode: bool, location: str):
    """Build config for smart analysis passes."""
    if flex_mode:
        for model_name, _ in get_smart_models(flex_mode=True):
            validate_flex_inference(model_name, location=location)

    return build_generation_config(temperature=temperature, thinking_budget=0, flex_mode=flex_mode)


def run_smart_analysis(
    dataset_name: str,
    limit_files: int,
    temperature: float,
    thinking_budget: int,
    similarity_threshold: int,
    flex_mode: bool = False,
    analysis_scope: str = None,
    recheck_problematic: bool = False,
    hf_token: str = None,
    progress=gr.Progress()
):
    from core.state import set_stop_requested, get_stop_requested
    set_stop_requested(False)
    set_analysis_running(True)
    
    global_results = get_global_results()
    
    # Robust type conversion for Gradio inputs
    limit_files = int(float(limit_files)) if limit_files else 0
    thinking_budget = int(float(thinking_budget)) if thinking_budget else 0
    similarity_threshold = int(float(similarity_threshold)) if similarity_threshold else 90
    temperature = float(temperature)
    analysis_scope = normalize_analysis_scope(
        analysis_scope,
        recheck_problematic=recheck_problematic,
    )

    models = get_smart_models(flex_mode=flex_mode)

    try:
        gemini_tool = GeminiIntegrator()
        gen_config = build_smart_generation_config(
            temperature=temperature,
            flex_mode=flex_mode,
            location=gemini_tool.location,
        )

        results = []
        
        # STEP 1: Initialization / First Pass
        step_desc = models[0][1]
        model_name = models[0][0]

        if analysis_scope == ANALYSIS_SCOPE_ALL or (
            analysis_scope in {ANALYSIS_SCOPE_PENDING, ANALYSIS_SCOPE_PENDING_OR_PROBLEMATIC}
            and not global_results
        ):
            results = _smart_fresh_first_pass(
                gemini_tool, model_name, step_desc, dataset_name,
                limit_files, analysis_scope, similarity_threshold, gen_config,
                hf_token, progress
            )
        else:
            results = _smart_recheck_first_pass(
                gemini_tool, model_name, step_desc, dataset_name,
                limit_files, analysis_scope, similarity_threshold, gen_config,
                hf_token, progress
            )

        # STEP 2-4: Iterative improvement
        base_progress = 0.25
        step_progress_size = 0.25

        for step_idx in range(1, len(models)):
            if get_stop_requested():
                print("🛑 Разумны аналіз спынены")
                break
            model_name = models[step_idx][0]
            step_desc = models[step_idx][1]

            # Find items that are STILL problematic AND not verified correct
            problematic_indices = [
                i for i, r in enumerate(results) 
                if r is not None and r.get('score', 0) < similarity_threshold 
                and r.get('verification_status') != 'correct'
            ]

            if not problematic_indices:
                progress(base_progress + step_idx * step_progress_size,
                         desc=f"{step_desc}: няма праблемных запісаў, прапускаем...")
                continue

            progress(base_progress + (step_idx - 1) * step_progress_size,
                     desc=f"{step_desc}: пераапрацоўка {len(problematic_indices)} праблемных запісаў...")

            for j, res_idx in enumerate(problematic_indices):
                if get_stop_requested():
                    break
                progress(base_progress + (step_idx - 1) * step_progress_size + (j + 1) / len(problematic_indices) * step_progress_size,
                         desc=f"{step_desc}: запіс {j+1}/{len(problematic_indices)}")

                result = results[res_idx]
                audio_data = result.get('audio_array')
                sampling_rate = result.get('sampling_rate')
                ref_text = result.get('ref_text', "")
                
                if audio_data is None or len(audio_data) == 0:
                    continue

                hyp_text = gemini_tool.transcribe_audio(model_name, audio_data, sampling_rate, config=gen_config)
                if is_transcription_error(hyp_text):
                    print(f"⚠️ Прапускаем (памылка распазнавання): {result.get('path')} | {hyp_text}")
                    continue
                score, norm_ref, norm_hyp = utils.calculate_similarity(ref_text, hyp_text)

                # Save model result
                if 'model_results' not in results[res_idx]:
                    results[res_idx]['model_results'] = {}
                
                results[res_idx]['model_results'][model_name] = {
                    "hyp_text": hyp_text,
                    "score": score,
                    "norm_ref": norm_ref,
                    "norm_hyp": norm_hyp
                }
                
                # Select best result
                best_model, best_result = select_best_model_result(
                    results[res_idx]['model_results'], 
                    similarity_threshold
                )
                
                if best_result and (best_result['score'] > result['score'] or best_result['score'] >= similarity_threshold):
                    new_status = "correct" if best_result['score'] >= similarity_threshold else "incorrect"
                    print(f"✅ UPDATE APPLIED [Idx={res_idx}]: {result.get('path')} | Best model: {best_model} | Score: {result['score']} -> {best_result['score']}")
                    results[res_idx].update({
                        "hyp_text": best_result['hyp_text'],
                        "score": best_result['score'],
                        "norm_ref": best_result['norm_ref'],
                        "norm_hyp": best_result['norm_hyp'],
                        "model_used": best_model,
                        "verification_status": new_status
                    })
                else:
                    print(f"⏭️ SKIP UPDATE [Idx={res_idx}]: Best score {best_result['score'] if best_result else 'N/A'} is not better than {result.get('score')} and not meeting threshold {similarity_threshold}")

            # Intermediate save after each step
            set_global_results(results)
            save_results_csv(dataset_name, auto_prefix=False)

        set_global_results(results)
        return generate_dashboard_outputs(similarity_threshold)

    except Exception as e:
        append_analysis_log(f"Smart Analysis failed: {e}")
        print(f"❌ Smart Analysis failed: {e}")
        save_results_csv(dataset_name, auto_prefix=True)
        raise gr.Error(f"Памылка: {e}")
    finally:
        set_analysis_running(False)
        save_results_csv(dataset_name, auto_prefix=True)


def _smart_recheck_first_pass(
    gemini_tool, model_name, step_desc, dataset_name,
    limit_files, analysis_scope, similarity_threshold, gen_config,
    hf_token, progress
):
    """First pass for recheck mode."""
    global_results = get_global_results()
    
    if not global_results:
        gr.Warning("Няма вынікаў для пераправеркі.")
        return []
    
    results = global_results
    
    problematic_indices = get_analysis_target_indices(
        results,
        analysis_scope,
        similarity_threshold,
        limit_files=limit_files,
    )
    
    if not problematic_indices:
        gr.Info("Няма праблемных файлаў для пераправеркі.")
        return results

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
        progress(0.03, desc=f"Датасет закэшаваны")
    
    # Build audio map by filename
    audio_map = {}
    for item in ds:
        path = item['audio']['path']
        if path:
            fname = os.path.basename(path)
            audio_map[fname] = item
            audio_map[path] = item

    progress(0.05, desc=f"{step_desc}: пераправерка {len(problematic_indices)} запісаў...")

    for j, res_idx in enumerate(problematic_indices):
        from core.state import get_stop_requested
        if get_stop_requested():
            break
        progress(0.05 + (j + 1) / len(problematic_indices) * 0.20, desc=f"{step_desc}: запіс {j+1}/{len(problematic_indices)}")

        result = results[res_idx]
        audio_data = result.get('audio_array')
        sampling_rate = result.get('sampling_rate')
        ref_text = result.get('ref_text', "")
        
        # If audio is missing, try to fetch from dataset
        if audio_data is None or len(audio_data) == 0:
            path = result.get('path', '')
            item = audio_map.get(path) or audio_map.get(os.path.basename(path))

            if not item:
                rec_id = result.get('id')
                if rec_id is not None:
                    try:
                        rec_id = int(rec_id)
                        if 0 <= rec_id < len(ds):
                            item = ds[rec_id]
                    except:
                        pass

            if item:
                audio_data = item['audio']['array']
                sampling_rate = item['audio']['sampling_rate']
                results[res_idx]['audio_array'] = audio_data
                results[res_idx]['sampling_rate'] = sampling_rate
            else:
                 print(f"Smart Analysis Recheck: Skipping index {res_idx}, path '{path}', id {result.get('id')}: Audio not found.")
                 continue

        hyp_text = gemini_tool.transcribe_audio(model_name, audio_data, sampling_rate, config=gen_config)
        if is_transcription_error(hyp_text):
            print(f"⚠️ Прапускаем (памылка распазнавання): {result.get('path')} | {hyp_text}")
            continue
        score, norm_ref, norm_hyp = utils.calculate_similarity(ref_text, hyp_text)

        print(f"🔄 Smart Updated (Step 1): {result.get('path')} | Score: {result.get('score')} -> {score} | Text: {hyp_text}")

        if 'model_results' not in results[res_idx]:
            results[res_idx]['model_results'] = {}
        
        results[res_idx]['model_results'][model_name] = {
            "hyp_text": hyp_text,
            "score": score,
            "norm_ref": norm_ref,
            "norm_hyp": norm_hyp
        }
        
        best_model, best_result = select_best_model_result(
            results[res_idx]['model_results'], 
            similarity_threshold
        )
        
        if best_result:
            results[res_idx].update({
                "hyp_text": best_result['hyp_text'],
                "score": best_result['score'],
                "norm_ref": best_result['norm_ref'],
                "norm_hyp": best_result['norm_hyp'],
                "model_used": best_model,
                "verification_status": "correct" if best_result['score'] >= similarity_threshold else "incorrect"
            })
        
        # Periodic save every 20 items
        if (j + 1) % 20 == 0:
            set_global_results(results)
            save_results_csv(dataset_name, auto_prefix=False)

    return results


def _smart_fresh_first_pass(
    gemini_tool, model_name, step_desc, dataset_name,
    limit_files, analysis_scope, similarity_threshold, gen_config,
    hf_token, progress
):
    """First pass for fresh analysis, resuming from pending files if they exist."""
    global_results = get_global_results()

    # Check for pending (unprocessed) files from a previous interrupted run
    pending_indices = [
        i for i, r in enumerate(global_results)
        if r is not None and r.get('verification_status') == 'pending'
    ]

    if analysis_scope == ANALYSIS_SCOPE_PENDING and pending_indices:
        print(f"▶️ Smart: аднаўленне — знойдзена {len(pending_indices)} неапрацаваных файлаў (pending)")
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
            progress(0.03, desc=f"Датасет закэшаваны")

        # Build audio map
        audio_map = {}
        for item in ds:
            path = item['audio']['path']
            if path:
                audio_map[os.path.basename(path)] = item
                audio_map[path] = item

        results = global_results
        progress(0.05, desc=f"{step_desc}: апрацоўка {len(pending_indices)} pending запісаў...")

        for j, idx in enumerate(pending_indices):
            from core.state import get_stop_requested
            if get_stop_requested():
                break
            progress(0.05 + (j + 1) / len(pending_indices) * 0.20, desc=f"{step_desc}: запіс {j+1}/{len(pending_indices)}")

            result = results[idx]
            audio_data = result.get('audio_array')
            sampling_rate = result.get('sampling_rate')
            ref_text = result.get('ref_text', "")

            if audio_data is None or (hasattr(audio_data, '__len__') and len(audio_data) == 0):
                path = result.get('path', '')
                item = audio_map.get(path) or audio_map.get(os.path.basename(path))
                if not item:
                    rec_id = result.get('id')
                    if rec_id is not None:
                        try:
                            rec_id = int(rec_id)
                            if 0 <= rec_id < len(ds):
                                item = ds[rec_id]
                        except:
                            pass
                if item:
                    audio_data = item['audio']['array']
                    sampling_rate = item['audio']['sampling_rate']
                    results[idx]['audio_array'] = audio_data
                    results[idx]['sampling_rate'] = sampling_rate
                else:
                    print(f"Smart Fresh Resume: Skipping index {idx}, audio not found.")
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

        return results

    # No pending files — full fresh run
    limit = int(limit_files) if limit_files > 0 else None

    cached_ds = get_cached_dataset(dataset_name, limit)
    if cached_ds is not None:
        progress(0, desc=f"Выкарыстоўваю закэшаваны датасет '{dataset_name}'...")
        ds = cached_ds
    else:
        progress(0, desc=f"Загрузка датасета '{dataset_name}'...")
        ds = utils.load_hf_dataset(dataset_name, limit=limit, hf_token=hf_token)
        cache_dataset(dataset_name, limit, ds)
        progress(0.05, desc=f"Датасет закэшаваны для паўторнага выкарыстання")

    results = []
    progress(0.05, desc=f"{step_desc}: апрацоўка ўсіх {len(ds)} запісаў...")

    for idx, item in enumerate(ds):
        from core.state import get_stop_requested
        if get_stop_requested():
            break
        progress(0.05 + (idx + 1) / len(ds) * 0.20, desc=f"{step_desc}: файл {idx+1}/{len(ds)}")

        audio_data = item['audio']['array']
        sampling_rate = item['audio']['sampling_rate']
        ref_text = item.get('sentence') or item.get('text') or item.get('transcription') or item.get('transcript') or ""

        hyp_text = gemini_tool.transcribe_audio(model_name, audio_data, sampling_rate, config=gen_config)
        if is_transcription_error(hyp_text):
            print(f"⚠️ Прапускаем (памылка распазнавання): {item['audio']['path']} | {hyp_text}")
            # Add a pending record so it can be retried later
            results.append({
                "id": idx,
                "path": item['audio']['path'],
                "ref_text": ref_text,
                "hyp_text": "",
                "score": 0,
                "norm_ref": "",
                "norm_hyp": "",
                "audio_array": audio_data,
                "sampling_rate": sampling_rate,
                "model_used": model_name,
                "verification_status": "pending",
                "model_results": {}
            })
            continue
        score, norm_ref, norm_hyp = utils.calculate_similarity(ref_text, hyp_text)

        results.append({
            "id": idx,
            "path": item['audio']['path'],
            "ref_text": ref_text,
            "hyp_text": hyp_text,
            "score": score,
            "norm_ref": norm_ref,
            "norm_hyp": norm_hyp,
            "audio_array": audio_data,
            "sampling_rate": sampling_rate,
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

    return results

