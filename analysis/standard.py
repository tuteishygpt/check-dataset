"""Standard analysis with batch mode support."""
import os
import time
import tempfile
import re
import gradio as gr
import soundfile as sf
from google import genai

import utils
from core.state import get_global_results, set_global_results
from core.cache import get_cached_dataset, cache_dataset
from core.comparison import select_best_model_result
from ui.dashboard import generate_dashboard_outputs
from ui.dashboard import generate_dashboard_outputs
from gemini_api import GeminiIntegrator, BatchTask, DEFAULT_TRANSCRIPTION_PROMPT
from hf_asr import is_hf_asr_model, get_hf_asr_client, HF_BATCH_SIZE
from analysis.import_export import save_results_csv


def sanitize_filename(name):
    """Sanitize string to be used as a filename."""
    if not name:
        return "results"
    # Replace non-alphanumeric with underscore
    s = re.sub(r'[^\w\s-]', '_', name).strip().lower()
    # Replace whitespace with underscore
    s = re.sub(r'[-\s]+', '_', s)
    return s


def run_analysis(
    api_key: str,
    dataset_name: str,
    model_name: str,
    limit_files: int,
    temperature: float,
    thinking_budget: int,
    similarity_threshold: int,
    batch_mode: bool = False,
    recheck_problematic: bool = False,
    hf_token: str = None,
    progress=gr.Progress()
):
    from core.state import set_stop_requested, get_stop_requested
    set_stop_requested(False)
    
    global_results = get_global_results()
    
    # Robust type conversion for Gradio inputs
    limit_files = int(float(limit_files)) if limit_files else 0
    thinking_budget = int(float(thinking_budget)) if thinking_budget else 0
    similarity_threshold = int(float(similarity_threshold)) if similarity_threshold else 90
    temperature = float(temperature)

    # ---------------------------------------------------------
    # HUGGING FACE ASR MODE
    # ---------------------------------------------------------
    if is_hf_asr_model(model_name):
        outputs = _run_hf_asr_analysis(
            model_name, dataset_name, limit_files,
            similarity_threshold, recheck_problematic,
            hf_token, progress
        )
        save_results_csv(dataset_name, auto_prefix=True)
        return outputs

    # For Gemini models, API key is required
    if not api_key:
        raise gr.Error("Калі ласка, увядзіце Gemini API ключ.")

    try:
        gemini_tool = GeminiIntegrator(api_key=api_key)

        config_args = {"temperature": temperature}
        use_thinking = "thinking" in model_name

        if use_thinking and thinking_budget > 0:
            config_args["thinking_config"] = {
                "include_thoughts": True,
                "budget_tokens": thinking_budget
            }

        gen_config = genai.types.GenerateContentConfig(**config_args)

        # ---------------------------------------------------------
        # BATCH MODE
        # ---------------------------------------------------------
        if batch_mode:
            outputs = _run_batch_analysis(
                gemini_tool, model_name, dataset_name, limit_files,
                similarity_threshold, recheck_problematic, 
                hf_token, progress
            )
        elif recheck_problematic:
            outputs = _run_recheck_analysis(
                gemini_tool, model_name, dataset_name, limit_files,
                similarity_threshold, gen_config, 
                hf_token, progress
            )
        else:
            outputs = _run_fresh_analysis(
                gemini_tool, model_name, dataset_name, limit_files,
                similarity_threshold, gen_config, 
                hf_token, progress
            )
            
        return outputs

    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        # Final save on error
        save_results_csv(dataset_name, auto_prefix=True)
        raise gr.Error(f"Памылка: {e}")
    finally:
        # Final save always (unless error already saved it with prefix)
        save_results_csv(dataset_name, auto_prefix=True)


def _run_batch_analysis(
    gemini_tool, model_name, dataset_name, limit_files,
    similarity_threshold, recheck_problematic, 
    hf_token, progress
):
    """Run batch processing mode."""
    global_results = get_global_results()
    ds = None
    
    # 1. Prepare Data
    if recheck_problematic:
        if not global_results:
            gr.Warning("Няма вынікаў для пераправеркі.")
            return generate_dashboard_outputs(similarity_threshold)
        
        # Load full dataset for audio fallback
        limit = None 
        cached_ds = get_cached_dataset(dataset_name, limit)
        if cached_ds:
            ds = cached_ds
        else:
            progress(0, desc=f"Загрузка датасета '{dataset_name}'...")
            ds = utils.load_hf_dataset(dataset_name, limit=limit)
            cache_dataset(dataset_name, limit, ds)
    else:
        limit = int(limit_files) if limit_files > 0 else None
        cached_ds = get_cached_dataset(dataset_name, limit)
        if cached_ds:
            ds = cached_ds
        else:
            progress(0, desc=f"Загрузка датасета '{dataset_name}'...")
            ds = utils.load_hf_dataset(dataset_name, limit=limit, hf_token=hf_token)
            cache_dataset(dataset_name, limit, ds)
        
        # Init results if fresh run
        progress(0.1, desc="Ініцыялізацыя спісу...")
        new_results = []
        for idx, item in enumerate(ds):
            ref_text = item.get('sentence') or item.get('text') or item.get('transcription') or item.get('transcript') or ""
            new_results.append({
                "id": idx,
                "path": item['audio']['path'],
                "ref_text": ref_text,
                "hyp_text": "",
                "score": 0,
                "audio_array": item['audio']['array'],
                "sampling_rate": item['audio']['sampling_rate'],
                "model_used": model_name,
                "verification_status": "pending"
            })
        set_global_results(new_results)
        global_results = get_global_results()

    # 2. Prepare Tasks
    tasks = []
    tmp_dir_obj = tempfile.TemporaryDirectory()
    tmp_dir = tmp_dir_obj.name
    
    def prepare_task(idx, row_data, audio_ref):
        key = f"task_{idx}"
        fpath = audio_ref['audio']['path']
        
        # Verify file existence or dump numpy to WAV
        if not fpath or not os.path.exists(fpath):
            audio_arr = audio_ref['audio']['array']
            sr = audio_ref['audio']['sampling_rate']
            if len(audio_arr) == 0:
                return None
            
            clean_name = sanitize_filename(f"audio_{idx}")
            dump_path = os.path.join(tmp_dir, f"{clean_name}.wav")
            sf.write(dump_path, audio_arr, int(sr), format='WAV')
            fpath = dump_path
        
        return BatchTask(key=key, path=fpath, mime_type="audio/wav")

    progress(0.2, desc="Падрыхтоўка задач для пакетнага рэжыму...")
    task_map_idx = {}  # task_key -> result_index
    
    if recheck_problematic:
        # Identification logic
        target_indices = [
             i for i, r in enumerate(global_results) 
             if r['score'] < similarity_threshold 
             and r.get('verification_status') != 'correct'
        ]
        if limit_files > 0:
            target_indices = target_indices[:limit_files]
        
        if not target_indices:
             gr.Info("Няма праблемных файлаў для пераправеркі.")
             try: tmp_dir_obj.cleanup() 
             except: pass
             return generate_dashboard_outputs(similarity_threshold)

        # Create DS Map
        ds_map = {}
        for di, d_item in enumerate(ds):
            p = d_item['audio']['path']
            if p:
                ds_map[p] = d_item
                ds_map[os.path.basename(p)] = d_item
            ds_map[di] = d_item

        for global_res_idx in target_indices:
            from core.state import get_stop_requested
            if get_stop_requested():
                break
            res = global_results[global_res_idx]
            path = res.get('path', '')
            
            # Try finding item
            item = ds_map.get(path) or ds_map.get(os.path.basename(path))
            if not item and res.get('id') is not None:
                 try: item = ds[int(res.get('id'))]
                 except: pass

            if item:
                t = prepare_task(global_res_idx, res, item)
                if t:
                    tasks.append(t)
                    task_map_idx[t.key] = global_res_idx
    else:
        # Tasks for all
        for idx, res in enumerate(global_results):
            from core.state import get_stop_requested
            if get_stop_requested():
                break
            item = ds[idx] 
            t = prepare_task(idx, res, item)
            if t:
                tasks.append(t)
                task_map_idx[t.key] = idx

    if not tasks:
        gr.Warning("Не знойдзена задач для выканання (магчыма, адсутнічае аўдыя).")
        try: tmp_dir_obj.cleanup() 
        except: pass
        return generate_dashboard_outputs(similarity_threshold)

    # 3. Execute Batch
    progress(0.3, desc=f"Запуск пакетнай апрацоўкі ({len(tasks)} файлаў). Гэта зойме час...")
    prompt = DEFAULT_TRANSCRIPTION_PROMPT
    
    try:
        batch_results = gemini_tool.run_batch(tasks, model_name, prompt)
    except Exception as e:
        try: tmp_dir_obj.cleanup() 
        except: pass
        raise gr.Error(f"Batch failed: {e}")
    
    progress(0.9, desc="Апрацоўка вынікаў...")
    
    # 4. Map Results
    for key, text in batch_results.items():
        if key in task_map_idx:
            idx = task_map_idx[key]
            if idx < len(global_results):
                ref_text = global_results[idx]['ref_text']
                score, norm_ref, norm_hyp = utils.calculate_similarity(ref_text, text)
                
                global_results[idx].update({
                    "hyp_text": text,
                    "score": score,
                    "norm_ref": norm_ref,
                    "norm_hyp": norm_hyp,
                    "verification_status": "correct" if score >= similarity_threshold else "incorrect",
                    "model_used": f"batch_{model_name}"
                })
                
                if 'model_results' not in global_results[idx]:
                    global_results[idx]['model_results'] = {}
                global_results[idx]['model_results'][model_name] = {
                    "hyp_text": text,
                    "score": score,
                    "norm_ref": norm_ref,
                    "norm_hyp": norm_hyp
                }

    try: tmp_dir_obj.cleanup() 
    except: pass
    
    return generate_dashboard_outputs(similarity_threshold)


def _run_recheck_analysis(
    gemini_tool, model_name, dataset_name, limit_files,
    similarity_threshold, gen_config, 
    hf_token, progress
):
    """Run recheck of problematic files."""
    global_results = get_global_results()
    
    if not global_results:
        gr.Warning("Няма вынікаў для пераправеркі.")
        return generate_dashboard_outputs(similarity_threshold)
    
    # Identify problematic records
    target_indices = [
        i for i, r in enumerate(global_results) 
        if r['score'] < similarity_threshold 
        and r.get('verification_status') != 'correct'
    ]
    
    if limit_files > 0:
        target_indices = target_indices[:limit_files]
    
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
    
    # Build audio map by filename
    audio_map = {}
    for item in ds:
        path = item['audio']['path']
        if path:
            fname = os.path.basename(path)
            audio_map[fname] = item
            audio_map[path] = item

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
        if audio_data is None or len(audio_data) == 0:
            path = result.get('path', '')
            item = audio_map.get(path) or audio_map.get(os.path.basename(path))
            
            # Fallback: try to find by ID if path lookup failed
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
                global_results[idx]['audio_array'] = audio_data
                global_results[idx]['sampling_rate'] = sampling_rate
            else:
                print(f"Problematic Recheck: Skipping index {idx}, path '{path}', id {result.get('id')}: Audio not found in dataset.")
                continue

        hyp_text = gemini_tool.transcribe_audio(model_name, audio_data, sampling_rate, config=gen_config)
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
    similarity_threshold, gen_config, 
    hf_token, progress
):
    """Run fresh analysis on all files."""
    limit = int(limit_files) if limit_files > 0 else None

    cached_ds = get_cached_dataset(dataset_name, limit)
    if cached_ds is not None:
        progress(0, desc=f"Выкарыстоўваю закэшаваны датасет '{dataset_name}'...")
        ds = cached_ds
    else:
        progress(0, desc=f"Загрузка датасета '{dataset_name}'...")
        ds = utils.load_hf_dataset(dataset_name, limit=limit, hf_token=hf_token)
        cache_dataset(dataset_name, limit, ds)
        progress(0.1, desc=f"Датасет закэшаваны для паўторнага выкарыстання")

    results = []

    for idx, item in enumerate(ds):
        from core.state import get_stop_requested
        if get_stop_requested():
            print("🛑 Аналіз спынены карыстальнікам")
            break
        progress((idx + 1) / len(ds), desc=f"Апрацоўка файла {idx+1}/{len(ds)}")

        audio_data = item['audio']['array']
        sampling_rate = item['audio']['sampling_rate']
        ref_text = item.get('sentence') or item.get('text') or item.get('transcription') or item.get('transcript') or ""

        hyp_text = gemini_tool.transcribe_audio(model_name, audio_data, sampling_rate, config=gen_config)
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
            set_global_results(results) # Update global state to save current results
            save_results_csv(dataset_name, auto_prefix=False)
    
    set_global_results(results)
    return generate_dashboard_outputs(similarity_threshold)


def _run_hf_asr_analysis(
    model_name: str,
    dataset_name: str,
    limit_files: int,
    similarity_threshold: int,
    recheck_problematic: bool,
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
    
    if recheck_problematic:
        return _run_hf_recheck_analysis(
            hf_client, model_name, dataset_name, limit_files,
            similarity_threshold, hf_token, progress
        )
    else:
        return _run_hf_fresh_analysis(
            hf_client, model_name, dataset_name, limit_files,
            similarity_threshold, hf_token, progress
        )


def _run_hf_fresh_analysis(
    hf_client, model_name, dataset_name, limit_files,
    similarity_threshold, hf_token, progress
):
    """Run fresh analysis using HF ASR with batch processing."""
    limit = int(limit_files) if limit_files > 0 else None
    
    cached_ds = get_cached_dataset(dataset_name, limit)
    if cached_ds is not None:
        progress(0, desc=f"Выкарыстоўваю закэшаваны датасет '{dataset_name}'...")
        ds = cached_ds
    else:
        progress(0, desc=f"Загрузка датасета '{dataset_name}'...")
        ds = utils.load_hf_dataset(dataset_name, limit=limit, hf_token=hf_token)
        cache_dataset(dataset_name, limit, ds)
        progress(0.1, desc=f"Датасет закэшаваны для паўторнага выкарыстання")
    
    # Pre-collect all items with their data
    all_items = []
    for idx, item in enumerate(ds):
        audio_data = item['audio']['array']
        sampling_rate = item['audio']['sampling_rate']
        ref_text = item.get('sentence') or item.get('text') or item.get('transcription') or item.get('transcript') or ""
        all_items.append({
            "idx": idx,
            "path": item['audio']['path'],
            "audio_data": audio_data,
            "sampling_rate": sampling_rate,
            "ref_text": ref_text
        })
    
    total_items = len(all_items)
    results = [None] * total_items  # Pre-allocate for correct ordering
    
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
            
            # Only record result if transcription was successful
            if hyp_text:
                score, norm_ref, norm_hyp = utils.calculate_similarity(ref_text, hyp_text)
                transcribed_count += 1
                
                results[idx] = {
                    "id": idx,
                    "path": item["path"],
                    "ref_text": ref_text,
                    "hyp_text": hyp_text,
                    "score": score,
                    "norm_ref": norm_ref,
                    "norm_hyp": norm_hyp,
                    "audio_array": item["audio_data"],
                    "sampling_rate": item["sampling_rate"],
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
    similarity_threshold, hf_token, progress
):
    """Run recheck of problematic files using HF ASR with batch processing."""
    global_results = get_global_results()
    
    if not global_results:
        gr.Warning("Няма вынікаў для пераправеркі.")
        return generate_dashboard_outputs(similarity_threshold)
    
    # Identify problematic records
    target_indices = [
        i for i, r in enumerate(global_results) 
        if r['score'] < similarity_threshold 
        and r.get('verification_status') != 'correct'
    ]
    
    if limit_files > 0:
        target_indices = target_indices[:limit_files]
    
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
    
    # Build audio map by filename
    audio_map = {}
    for item in ds:
        path = item['audio']['path']
        if path:
            fname = os.path.basename(path)
            audio_map[fname] = item
            audio_map[path] = item
    
    # Collect all items to process with their audio data
    items_to_process = []
    for idx in target_indices:
        result = global_results[idx]
        audio_data = result.get('audio_array')
        sampling_rate = result.get('sampling_rate')
        ref_text = result.get('ref_text', "")
        
        # If audio is missing, try to fetch from dataset
        if audio_data is None or (hasattr(audio_data, '__len__') and len(audio_data) == 0):
            path = result.get('path', '')
            item = audio_map.get(path) or audio_map.get(os.path.basename(path))
            
            # Fallback: try to find by ID if path lookup failed
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
                global_results[idx]['audio_array'] = audio_data
                global_results[idx]['sampling_rate'] = sampling_rate
            else:
                print(f"HF Recheck: Skipping index {idx}, path '{path}': Audio not found.")
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
            
            if not hyp_text:
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
