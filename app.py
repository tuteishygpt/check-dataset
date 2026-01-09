import gradio as gr
import json
import pandas as pd
from google import genai
import os
from dotenv import load_dotenv
import utils
import io
import soundfile as sf
import numpy as np
import hashlib
import base64
import html

# Load environment variables
load_dotenv()

# Global variable to store results for audio playback
global_results = []

# Cache for downloaded datasets
dataset_cache = {}


def get_dataset_cache_key(dataset_name: str, limit: int) -> str:
    """Generate a cache key for the dataset."""
    return hashlib.md5(f"{dataset_name}:{limit}".encode()).hexdigest()


def get_cached_dataset(dataset_name: str, limit: int):
    """Get cached dataset if available."""
    cache_key = get_dataset_cache_key(dataset_name, limit)
    return dataset_cache.get(cache_key)


def cache_dataset(dataset_name: str, limit: int, data):
    """Cache the downloaded dataset."""
    cache_key = get_dataset_cache_key(dataset_name, limit)
    dataset_cache[cache_key] = data


def array_to_b64_audio(audio_array, sampling_rate):
    """Convert numpy array audio to base64 encoded HTML audio tag."""
    if audio_array is None or len(audio_array) == 0:
        return '<div style="width: 100%; margin-top: 10px; height: 36px; border-radius: 8px; background: rgba(0,0,0,0.2); display: flex; align-items: center; justify-content: center; color: #64748b;">🔇 Аўдыя недаступна</div>'

    buffer = io.BytesIO()
    # Ensure sampling_rate is an integer and handle potential NaNs from Pandas
    if pd.notnull(sampling_rate):
        sr = int(float(sampling_rate))
    else:
        sr = 16000
    sf.write(buffer, audio_array, sr, format='WAV')
    buffer.seek(0)
    b64_data = base64.b64encode(buffer.read()).decode('utf-8')
    return f'<audio controls src="data:audio/wav;base64,{b64_data}" style="width: 100%; margin-top: 10px; height: 36px; border-radius: 8px;"></audio>'


def _e(x) -> str:
    """HTML-escape helper."""
    return html.escape("" if x is None else str(x), quote=True)


def generate_dashboard_outputs(similarity_threshold: int):
    """
    Generates the HTML/DF outputs for the dashboard based on global_results.
    Refactored to be used by all analysis functions.
    """
    global global_results

    df = pd.DataFrame(global_results)

    if df.empty:
        return "", "", pd.DataFrame()

    # Ensure columns exist
    if 'verification_status' not in df.columns:
        df['verification_status'] = None
    if 'model_used' not in df.columns:
        df['model_used'] = "unknown"

    # Statistics
    total_files = len(df)

    # Problematic: below threshold AND NOT verified as correct
    flagged_mask = (df['score'] < similarity_threshold) & (df['verification_status'] != 'correct')
    flagged_count = int(flagged_mask.sum())
    avg_score = df['score'].mean() if len(df) > 0 else 0

    # Model stats
    model_stats = ""
    if 'model_used' in df.columns:
        model_counts = df['model_used'].value_counts().to_dict()
        model_stats_str = " | ".join([f"{_e(m)}: {c}" for m, c in model_counts.items()])
        model_stats = f"""
        <div style="background: rgba(30, 41, 59, 0.8); padding: 15px; border-radius: 12px; margin-bottom: 20px;">
            <p style="color: #94a3b8; margin: 0;">🤖 <strong>Мадэлі:</strong> {model_stats_str}</p>
        </div>
        """

    stats_html = f"""
    <div style="display: flex; gap: 20px; margin-bottom: 20px;">
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    padding: 20px; border-radius: 12px; text-align: center; flex: 1; color: white;">
            <h3 style="margin: 0; font-size: 2em;">{total_files}</h3>
            <p style="margin: 5px 0 0 0; opacity: 0.9;">📁 Усяго файлаў</p>
        </div>
        <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                    padding: 20px; border-radius: 12px; text-align: center; flex: 1; color: white;">
            <h3 style="margin: 0; font-size: 2em;">{flagged_count}</h3>
            <p style="margin: 5px 0 0 0; opacity: 0.9;">🚩 Праблемных</p>
        </div>
        <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            padding: 20px; border-radius: 12px; text-align: center; flex: 1; color: white;">
            <h3 style="margin: 0; font-size: 2em;">{avg_score:.1f}%</h3>
            <p style="margin: 5px 0 0 0; opacity: 0.9;">📊 Сярэдні скор</p>
        </div>
    </div>
    {model_stats}
    """

    # Flagged items HTML
    flagged_df = df[flagged_mask].sort_values(by="score")

    if flagged_df.empty:
        flagged_html = """
        <div style="background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
                    padding: 30px; border-radius: 12px; text-align: center;">
            <h3 style="color: #2d3748; margin: 0;">✅ Усё добра!</h3>
            <p style="color: #4a5568; margin: 10px 0 0 0;">Файлаў ніжэй парогу не знойдзена (або ўсе правераны).</p>
        </div>
        """
    else:
        flagged_html = ""
        for _, row in flagged_df.iterrows():
            rid = int(row['id']) if pd.notnull(row.get('id')) else -1
            score = float(row['score']) if pd.notnull(row.get('score')) else 0.0
            score_int = int(round(score))
            score_color = "#f5576c" if score < 50 else "#fbbf24" if score < 75 else "#34d399"

            audio_html = array_to_b64_audio(row.get('audio_array'), row.get('sampling_rate'))

            model_used = row.get('model_used', 'unknown')
            model_badge = "🖐️ Ручная праверка" if model_used == 'manual' else _e(model_used)

            flagged_html += f"""
            <div style="background: #1e293b; border-radius: 12px; padding: 20px; margin-bottom: 15px;
                        border-left: 4px solid {score_color};">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                    <div style="display: flex; flex-direction: column;">
                        <span style="color: #e2e8f0; font-weight: bold;">📄 {_e(row.get('path'))}</span>
                        <div style="margin-top: 5px;">
                             <span style="background: #475569; color: #e2e8f0; padding: 3px 8px;
                                          border-radius: 10px; font-size: 0.8em; margin-right: 8px;">🤖 {model_badge}</span>
                        </div>
                    </div>
                    <span style="background: {score_color}; color: white; padding: 5px 12px;
                                 border-radius: 20px; font-weight: bold;">{score_int}%</span>
                </div>
                <div style="background: #0f172a; border-radius: 8px; padding: 15px; margin-bottom: 10px;">
                    <p style="color: #94a3b8; margin: 0 0 5px 0; font-size: 0.85em;">📝 Арыгінал:</p>
                    <p style="color: #f1f5f9; margin: 0; font-family: monospace;">{_e(row.get('ref_text'))}</p>
                </div>
                <div style="background: #0f172a; border-radius: 8px; padding: 15px; margin-bottom: 10px;">
                    <p style="color: #94a3b8; margin: 0 0 5px 0; font-size: 0.85em;">🎤 Распазнана:</p>
                    <p style="color: #f1f5f9; margin: 0; font-family: monospace;">{_e(row.get('hyp_text'))}</p>
                </div>
                <details style="color: #94a3b8; margin-bottom: 10px;">
                    <summary style="cursor: pointer; color: #60a5fa;">🔍 Нармалізаваны тэкст</summary>
                    <div style="background: #0f172a; border-radius: 8px; padding: 10px; margin-top: 10px;">
                        <p style="margin: 5px 0;"><strong>Ref:</strong> {_e(row.get('norm_ref'))}</p>
                        <p style="margin: 5px 0;"><strong>Hyp:</strong> {_e(row.get('norm_hyp'))}</p>
                    </div>
                </details>
                {audio_html}
                <div style="display: flex; gap: 10px; margin-top: 15px; justify-content: flex-end;">
                    <button type="button" onclick="verifyRecord({rid}, 'correct')" class="verify-btn correct-btn">✅ Правільна</button>
                    <button type="button" onclick="verifyRecord({rid}, 'incorrect')" class="verify-btn incorrect-btn">❌ Няправільна</button>
                </div>
            </div>
            """

    # Add minimized rows for manually verified items
    manual_mask = (df['verification_status'].notnull()) & (df['model_used'] == 'manual')
    manual_df = df[manual_mask].sort_values(by="id", ascending=False).head(5)

    if not manual_df.empty:
        flagged_html += """<h4 style="color: #94a3b8; margin: 20px 0 10px 0;">🕒 Апошнія правераныя:</h4>"""
        for _, row in manual_df.iterrows():
            status_icon = "✅" if row['verification_status'] == 'correct' else "❌"
            status_color = "#10b981" if row['verification_status'] == 'correct' else "#ef4444"
            flagged_html += f"""
            <div style="background: rgba(30, 41, 59, 0.4); border-radius: 8px; padding: 10px 15px;
                        margin-bottom: 8px; display: flex; justify-content: space-between; align-items: center;
                        border-left: 3px solid {status_color};">
                <span style="color: #cbd5e1; font-size: 0.9em;">{_e(row.get('path'))}</span>
                <span style="color: {status_color}; font-weight: bold; font-size: 0.9em;">
                    {status_icon} {_e(row.get('verification_status'))} (Score: {int(round(float(row.get('score', 0))))}%)
                </span>
            </div>
            """

    # Full table
    cols = ['path', 'score', 'model_used', 'verification_status', 'ref_text', 'hyp_text']

    display_df = df.copy()
    if 'verification_status' not in display_df.columns:
        display_df['verification_status'] = None

    def map_status(x):
        if x == 'correct':
            return "✅"
        if x == 'incorrect':
            return "❌"
        return ""

    display_df['verification_status'] = display_df['verification_status'].apply(map_status)
    table_df = display_df[cols].sort_values(by="score")

    return stats_html, flagged_html, table_df


def run_analysis(
    api_key: str,
    dataset_name: str,
    model_name: str,
    limit_files: int,
    temperature: float,
    thinking_budget: int,
    similarity_threshold: int,
    recheck_problematic: bool = False,
    progress=gr.Progress()
):
    global global_results
    
    # Robust type conversion for Gradio inputs
    limit_files = int(float(limit_files)) if limit_files else 0
    thinking_budget = int(float(thinking_budget)) if thinking_budget else 0
    similarity_threshold = int(float(similarity_threshold)) if similarity_threshold else 90
    temperature = float(temperature)

    if not api_key:
        raise gr.Error("Калі ласка, увядзіце Gemini API ключ.")

    try:
        client = genai.Client(api_key=api_key)

        config_args = {"temperature": temperature}
        use_thinking = "thinking" in model_name

        if use_thinking and thinking_budget > 0:
            config_args["thinking_config"] = {
                "include_thoughts": True,
                "budget_tokens": thinking_budget
            }

        gen_config = genai.types.GenerateContentConfig(**config_args)

        if recheck_problematic:
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
            limit = None # Always load full dataset for rechecking to ensure we find matches
            cached_ds = get_cached_dataset(dataset_name, limit)
            if cached_ds is not None:
                progress(0, desc=f"Выкарыстоўваю закэшаваны датасет '{dataset_name}'...")
                ds = cached_ds
            else:
                progress(0, desc=f"Загрузка датасета '{dataset_name}'...")
                ds = utils.load_hf_dataset(dataset_name, limit=limit)
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
                progress(0.1 + (j + 1) / len(target_indices) * 0.9, desc=f"Праверка {j+1}/{len(target_indices)}")
                
                result = global_results[idx]
                audio_data = result.get('audio_array')
                sampling_rate = result.get('sampling_rate')
                ref_text = result.get('ref_text', "")
                
                # If audio is missing, try to fetch from dataset
                if audio_data is None or len(audio_data) == 0:
                    path = result.get('path', '')
                    item = audio_map.get(path) or audio_map.get(os.path.basename(path))
                    if item:
                        audio_data = item['audio']['array']
                        sampling_rate = item['audio']['sampling_rate']
                        # Also update the global_results with audio for future use
                        global_results[idx]['audio_array'] = audio_data
                        global_results[idx]['sampling_rate'] = sampling_rate
                    else:
                        continue  # Skip if audio still not found

                hyp_text = utils.transcribe_audio(client, model_name, audio_data, sampling_rate, config=gen_config)
                score, norm_ref, norm_hyp = utils.calculate_similarity(ref_text, hyp_text)

                global_results[idx].update({
                    "hyp_text": hyp_text,
                    "score": score,
                    "norm_ref": norm_ref,
                    "norm_hyp": norm_hyp,
                    "model_used": model_name,
                    "verification_status": "correct" if score >= similarity_threshold else "incorrect"
                })

        else:
            limit = int(limit_files) if limit_files > 0 else None

            cached_ds = get_cached_dataset(dataset_name, limit)
            if cached_ds is not None:
                progress(0, desc=f"Выкарыстоўваю закэшаваны датасет '{dataset_name}'...")
                ds = cached_ds
            else:
                progress(0, desc=f"Загрузка датасета '{dataset_name}'...")
                ds = utils.load_hf_dataset(dataset_name, limit=limit)
                cache_dataset(dataset_name, limit, ds)
                progress(0.1, desc=f"Датасет закэшаваны для паўторнага выкарыстання")

            results = []

            for idx, item in enumerate(ds):
                progress((idx + 1) / len(ds), desc=f"Апрацоўка файла {idx+1}/{len(ds)}")

                audio_data = item['audio']['array']
                sampling_rate = item['audio']['sampling_rate']
                ref_text = item.get('sentence') or item.get('text') or item.get('transcription') or item.get('transcript') or ""

                hyp_text = utils.transcribe_audio(client, model_name, audio_data, sampling_rate, config=gen_config)
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
                    "verification_status": "correct" if score >= similarity_threshold else "incorrect"
                })
            
            global_results = results

        return generate_dashboard_outputs(similarity_threshold)

    except Exception as e:
        raise gr.Error(f"Памылка: {e}")


def run_smart_analysis(
    api_key: str,
    dataset_name: str,
    limit_files: int,
    temperature: float,
    thinking_budget: int,
    similarity_threshold: int,
    recheck_problematic: bool = False,
    progress=gr.Progress()
):
    global global_results
    
    # Robust type conversion for Gradio inputs
    limit_files = int(float(limit_files)) if limit_files else 0
    thinking_budget = int(float(thinking_budget)) if thinking_budget else 0
    similarity_threshold = int(float(similarity_threshold)) if similarity_threshold else 90
    temperature = float(temperature)

    if not api_key:
        raise gr.Error("Калі ласка, увядзіце Gemini API ключ.")

    models = [
        ("gemini-2.5-flash-lite", "Этап 1/4: Flash-Lite (першы праход)"),
        ("gemini-2.5-flash-lite", "Этап 2/4: Flash-Lite (другі праход)"),
        ("gemini-2.5-flash", "Этап 3/4: Flash"),
        ("gemini-3-flash-preview", "Этап 4/4: Gemini-3-Flash"),
    ]

    try:
        client = genai.Client(api_key=api_key)

        config_args = {"temperature": temperature}
        gen_config = genai.types.GenerateContentConfig(**config_args)

        results = []
        
        # STEP 1: Initialization / First Pass
        step_desc = models[0][1]
        model_name = models[0][0]

        if recheck_problematic:
            if not global_results:
                gr.Warning("Няма вынікаў для пераправеркі.")
                return generate_dashboard_outputs(similarity_threshold)
            
            results = global_results # Work on the global list directly/by reference
            
            # Identify start set: only problematic items
            problematic_indices = [
                i for i, r in enumerate(results) 
                if r['score'] < similarity_threshold 
                and r.get('verification_status') != 'correct'
            ]
            
            if limit_files > 0:
                problematic_indices = problematic_indices[:limit_files]
            
            if not problematic_indices:
                gr.Info("Няма праблемных файлаў для пераправеркі.")
                return generate_dashboard_outputs(similarity_threshold)

            # Load dataset to get audio for files that might be missing it
            limit = None # Always load full dataset for rechecking to ensure we find matches
            cached_ds = get_cached_dataset(dataset_name, limit)
            if cached_ds is not None:
                progress(0, desc=f"Выкарыстоўваю закэшаваны датасет '{dataset_name}'...")
                ds = cached_ds
            else:
                progress(0, desc=f"Загрузка датасета '{dataset_name}'...")
                ds = utils.load_hf_dataset(dataset_name, limit=limit)
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
                progress(0.05 + (j + 1) / len(problematic_indices) * 0.20, desc=f"{step_desc}: запіс {j+1}/{len(problematic_indices)}")

                result = results[res_idx]
                audio_data = result.get('audio_array')
                sampling_rate = result.get('sampling_rate')
                ref_text = result.get('ref_text', "")
                
                # If audio is missing, try to fetch from dataset
                if audio_data is None or len(audio_data) == 0:
                    path = result.get('path', '')
                    item = audio_map.get(path) or audio_map.get(os.path.basename(path))
                    if item:
                        audio_data = item['audio']['array']
                        sampling_rate = item['audio']['sampling_rate']
                        # Also update with audio for future use
                        results[res_idx]['audio_array'] = audio_data
                        results[res_idx]['sampling_rate'] = sampling_rate
                    else:
                        continue  # Skip if audio still not found

                hyp_text = utils.transcribe_audio(client, model_name, audio_data, sampling_rate, config=gen_config)
                score, norm_ref, norm_hyp = utils.calculate_similarity(ref_text, hyp_text)

                results[res_idx].update({
                    "hyp_text": hyp_text,
                    "score": score,
                    "norm_ref": norm_ref,
                    "norm_hyp": norm_hyp,
                    "model_used": model_name,
                    "verification_status": "correct" if score >= similarity_threshold else "incorrect"
                })

        else:
            # Standard logic: load dataset and process all
            limit = int(limit_files) if limit_files > 0 else None

            cached_ds = get_cached_dataset(dataset_name, limit)
            if cached_ds is not None:
                progress(0, desc=f"Выкарыстоўваю закэшаваны датасет '{dataset_name}'...")
                ds = cached_ds
            else:
                progress(0, desc=f"Загрузка датасета '{dataset_name}'...")
                ds = utils.load_hf_dataset(dataset_name, limit=limit)
                cache_dataset(dataset_name, limit, ds)
                progress(0.05, desc=f"Датасет закэшаваны для паўторнага выкарыстання")

            progress(0.05, desc=f"{step_desc}: апрацоўка ўсіх {len(ds)} запісаў...")

            for idx, item in enumerate(ds):
                progress(0.05 + (idx + 1) / len(ds) * 0.20, desc=f"{step_desc}: файл {idx+1}/{len(ds)}")

                audio_data = item['audio']['array']
                sampling_rate = item['audio']['sampling_rate']
                ref_text = item.get('sentence') or item.get('text') or item.get('transcription') or item.get('transcript') or ""

                hyp_text = utils.transcribe_audio(client, model_name, audio_data, sampling_rate, config=gen_config)
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
                    "verification_status": "correct" if score >= similarity_threshold else "incorrect"
                })

        # STEP 2-4: Iterative improvement
        # Logic remains mostly the same, but we should respect 'verification_status' != 'correct'
        base_progress = 0.25
        step_progress_size = 0.25

        for step_idx in range(1, len(models)):
            model_name = models[step_idx][0]
            step_desc = models[step_idx][1]

            # Find items that are STILL problematic AND not verified correct
            problematic_indices = [
                i for i, r in enumerate(results) 
                if r['score'] < similarity_threshold 
                and r.get('verification_status') != 'correct'
            ]

            if not problematic_indices:
                progress(base_progress + step_idx * step_progress_size,
                         desc=f"{step_desc}: няма праблемных запісаў, прапускаем...")
                continue

            progress(base_progress + (step_idx - 1) * step_progress_size,
                     desc=f"{step_desc}: пераапрацоўка {len(problematic_indices)} праблемных запісаў...")

            for j, res_idx in enumerate(problematic_indices):
                progress(base_progress + (step_idx - 1) * step_progress_size + (j + 1) / len(problematic_indices) * step_progress_size,
                         desc=f"{step_desc}: запіс {j+1}/{len(problematic_indices)}")

                result = results[res_idx]
                audio_data = result.get('audio_array') # use get()
                sampling_rate = result.get('sampling_rate')
                ref_text = result.get('ref_text', "")
                
                if audio_data is None or len(audio_data) == 0:
                    continue

                hyp_text = utils.transcribe_audio(client, model_name, audio_data, sampling_rate, config=gen_config)
                score, norm_ref, norm_hyp = utils.calculate_similarity(ref_text, hyp_text)

                if score > result['score']:
                    new_status = "correct" if score >= similarity_threshold else "incorrect"
                    results[res_idx].update({
                        "hyp_text": hyp_text,
                        "score": score,
                        "norm_ref": norm_ref,
                        "norm_hyp": norm_hyp,
                        "model_used": model_name,
                        "verification_status": new_status
                    })

        global_results = results
        return generate_dashboard_outputs(similarity_threshold)

    except Exception as e:
        raise gr.Error(f"Памылка: {e}")





def get_audio_for_row(row_index: int):
    global global_results
    if row_index < 0 or row_index >= len(global_results):
        return None

    row = global_results[row_index]
    if row.get('audio_array') is None or len(row.get('audio_array')) == 0:
        return None

    buffer = io.BytesIO()
    sr = int(float(row['sampling_rate'])) if pd.notnull(row.get('sampling_rate')) else 16000
    sf.write(buffer, row['audio_array'], sr, format='WAV')
    buffer.seek(0)
    return (sr, np.array(row['audio_array']))


def update_thinking_visibility(model_name: str):
    return gr.update(visible="thinking" in model_name)


def clear_cache():
    global dataset_cache
    count = len(dataset_cache)
    dataset_cache.clear()
    return f"<p style='color: #34d399;'>✅ Кэш ачышчаны ({count} датасет(аў) выдалена)</p>"


def get_cache_status():
    if not dataset_cache:
        return "<p style='color: #94a3b8;'>📭 Кэш пусты</p>"

    total_items = 0
    for key in dataset_cache:
        total_items += len(dataset_cache[key])

    return f"<p style='color: #60a5fa;'>📦 Закэшавана: {len(dataset_cache)} датасет(аў), {total_items} элементаў</p>"


def import_csv_analysis(file_obj, similarity_threshold, dataset_name, limit_files):
    global global_results
    
    # Robust type conversion
    limit_files = int(float(limit_files)) if limit_files else 0
    similarity_threshold = int(float(similarity_threshold)) if similarity_threshold else 90

    if file_obj is None:
        return generate_dashboard_outputs(similarity_threshold)
    
    try:
        # Load CSV
        df = pd.read_csv(file_obj.name)
        
        # Check required columns (flexible check)
        # The user provided: idx,file_name,score_%,ref,hyp
        # We should support this specific format.
        
        # Map columns if needed
        col_map = {
            'score_%': 'score',
            'file_name': 'path',
            'ref': 'ref_text',
            'hyp': 'hyp_text',
            'idx': 'id'
        }
        
        # Check if we have the specific columns requested
        user_cols = ['idx', 'file_name', 'score_%', 'ref', 'hyp']
        # If all user_cols are present, we are good. If not, we try to survive.
        
        # Try to load audio from dataset cache if available
        audio_map = {}
        limit = None # Ignore limit when importing to find all matching audio
        cached_ds = get_cached_dataset(dataset_name, limit)
        
        if cached_ds:
             for item in cached_ds:
                 path = item['audio']['path']
                 if path:
                     fname = os.path.basename(path)
                     audio_map[fname] = item
                     audio_map[path] = item # Store both full path and basename

        results = []
        for idx, row in df.iterrows():
            # Get values using the specific column names
            fname = str(row.get('file_name', row.get('path', '')))
            ref = str(row.get('ref', row.get('ref_text', '')))
            hyp = str(row.get('hyp', row.get('hyp_text', '')))
            
            # Handle score: convert "91.67" or "91,67" to float
            score_val = row.get('score_%', row.get('score', 0))
            if isinstance(score_val, str):
                score_val = score_val.replace(',', '.')
            try:
                score = float(score_val)
            except:
                score = 0.0
            
            row_id = row.get('idx', row.get('id', idx))

            # Find audio
            audio_array = None
            sampling_rate = None
            
            # Try exact match or basename match
            item = audio_map.get(fname)
            if not item and fname:
                # Try basename if fname looks like a path
                item = audio_map.get(os.path.basename(fname))
            
            if item:
                 audio_array = item['audio']['array']
                 sampling_rate = item['audio']['sampling_rate']

            # Normalize using utils
            norm_ref = utils.normalize_text(ref)
            norm_hyp = utils.normalize_text(hyp)

            results.append({
                "id": int(row_id) if isinstance(row_id, (int, float, str)) and str(row_id).isdigit() else idx,
                "path": fname,
                "ref_text": ref,
                "hyp_text": hyp,
                "score": score,
                "norm_ref": norm_ref,
                "norm_hyp": norm_hyp,
                "audio_array": audio_array,
                "sampling_rate": sampling_rate,
                "model_used": "imported_csv",
                "verification_status": "correct" if score >= similarity_threshold else "incorrect"
            })
            
        global_results = results
        return generate_dashboard_outputs(similarity_threshold)
        
    except Exception as e:
        raise gr.Error(f"Памылка імпарту: {e}")


def _find_index_by_id(record_id: int):
    """Find index in global_results by record['id'] (not by list position)."""
    global global_results
    for i, r in enumerate(global_results):
        if r.get("id") == record_id:
            return i
    return None


def verify_action(data_str, similarity_threshold):
    """
    Handles verification button click.
    Expects JSON like: {"id": 12, "status": "correct", "ts": 123456}
    """
    global global_results
    similarity_threshold = int(float(similarity_threshold)) if similarity_threshold else 90

    # Always return a refreshed dashboard even if parsing fails
    if not data_str:
        return generate_dashboard_outputs(similarity_threshold)

    try:
        data = json.loads(data_str)
        record_id = data.get('id')
        status = data.get('status')

        if record_id is None or status not in ("correct", "incorrect"):
            return generate_dashboard_outputs(similarity_threshold)

        idx = _find_index_by_id(int(record_id))
        if idx is None:
            return generate_dashboard_outputs(similarity_threshold)

        global_results[idx]['verification_status'] = status
        global_results[idx]['model_used'] = 'manual'

        # Persist to CSV (best effort)
        try:
            save_df = pd.DataFrame(global_results)
            save_df.to_csv("check_dataset_results.csv", index=False)
        except Exception as e:
            print(f"Error saving to CSV: {e}")

        return generate_dashboard_outputs(similarity_threshold)

    except Exception as e:
        print(f"Error in verify_action: {e}")
        return generate_dashboard_outputs(similarity_threshold)


def save_results_csv():
    global global_results
    if not global_results:
        return None

    try:
        df = pd.DataFrame(global_results)
        cols_to_exclude = ['audio_array', 'sampling_rate']
        valid_cols = [c for c in df.columns if c not in cols_to_exclude]
        df_export = df[valid_cols]

        filename = "check_dataset_results.csv"
        df_export.to_csv(filename, index=False)
        return filename
    except Exception as e:
        print(f"Error creating CSV: {e}")
        return None


# CSS for dark theme and modern styling
custom_css = """
.gradio-container {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%) !important;
    min-height: 100vh;
}

.dark {
    --body-background-fill: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
}

.main-title {
    text-align: center;
    background: linear-gradient(90deg, #667eea, #764ba2, #f093fb);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 2.5em;
    font-weight: bold;
    margin-bottom: 10px;
}

.subtitle {
    text-align: center;
    color: #94a3b8;
    margin-bottom: 30px;
}

.settings-panel {
    background: rgba(30, 41, 59, 0.8) !important;
    border-radius: 16px !important;
    padding: 20px !important;
    backdrop-filter: blur(10px);
}

.results-panel {
    background: rgba(30, 41, 59, 0.6) !important;
    border-radius: 16px !important;
    padding: 20px !important;
}

.primary-btn {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    font-size: 1.1em !important;
    padding: 15px 30px !important;
    border-radius: 12px !important;
    transition: transform 0.2s, box-shadow 0.2s !important;
}

.primary-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 10px 40px rgba(102, 126, 234, 0.4) !important;
}

.section-title {
    color: #e2e8f0;
    font-size: 1.3em;
    margin: 20px 0 15px 0;
    display: flex;
    align-items: center;
    gap: 10px;
}

.verify-btn {
    padding: 8px 16px;
    border-radius: 8px;
    border: none;
    cursor: pointer;
    font-weight: bold;
    color: white;
    transition: transform 0.1s;
}

.verify-btn:hover {
    transform: scale(1.05);
}

.correct-btn { background: #10b981; }
.correct-btn:hover { background: #059669; }

.incorrect-btn { background: #ef4444; }
.incorrect-btn:hover { background: #dc2626; }

footer { display: none !important; }

/* Keep components rendered but out of view */
#verification_data_input, #verification_trigger_btn {
    position: fixed !important;
    top: -10000px !important;
    left: -10000px !important;
    opacity: 0 !important;
    width: 1px !important;
    height: 1px !important;
    pointer-events: none !important;
    z-index: 1 !important;
}
"""

# JS injected into <head> (reliable execution)
head_js = """
<script>
(function () {
  function gradioRoot() {
    const app = document.querySelector("gradio-app");
    if (app && app.shadowRoot) return app.shadowRoot;
    return document;
  }

  function qs(sel) {
    return gradioRoot().querySelector(sel);
  }

  function getValueEl(containerId) {
    const c = qs("#" + containerId);
    if (!c) return null;
    return c.querySelector("textarea, input");
  }

  function clickGradioButton(containerId) {
    const c = qs("#" + containerId);
    if (!c) return false;
    const btn = c.querySelector("button") || c; // IMPORTANT: click real <button> if wrapper <div> has the id
    btn.click();
    return true;
  }

  window.verifyRecord = function (id, status) {
    try {
      const el = getValueEl("verification_data_input");
      if (!el) {
        console.error("verification_data_input not found");
        return;
      }

      const payload = JSON.stringify({ id: id, status: status, ts: Date.now() });

      const proto = (el.tagName === "TEXTAREA")
        ? window.HTMLTextAreaElement.prototype
        : window.HTMLInputElement.prototype;

      const desc = Object.getOwnPropertyDescriptor(proto, "value");
      const setter = desc && desc.set;

      if (setter) setter.call(el, payload);
      else el.value = payload;

      el.dispatchEvent(new Event("input", { bubbles: true }));
      el.dispatchEvent(new Event("change", { bubbles: true }));

      setTimeout(() => {
        const ok = clickGradioButton("verification_trigger_btn");
        if (!ok) console.error("verification_trigger_btn not found");
      }, 0);

    } catch (e) {
      console.error("verifyRecord error:", e);
    }
  };

  // localStorage API key persistence
  function setupApiKeySaver() {
    const el = getValueEl("api_key_input");
    if (!el) return false;

    const saved = localStorage.getItem("gemini_api_key");
    if (saved && !el.value) {
      el.value = saved;
      el.dispatchEvent(new Event("input", { bubbles: true }));
      el.dispatchEvent(new Event("change", { bubbles: true }));
    }

    if (!el.__geminiSaverAttached) {
      el.addEventListener("blur", function () {
        if (this.value && this.value.length > 10) {
          localStorage.setItem("gemini_api_key", this.value);
        }
      });
      el.__geminiSaverAttached = true;
    }
    return true;
  }

  // retry a few times because Gradio renders async
  let tries = 0;
  const timer = setInterval(() => {
    tries += 1;
    const ok = setupApiKeySaver();
    if (ok || tries >= 40) clearInterval(timer);
  }, 250);

})();
</script>
"""

# Build the interface
with gr.Blocks(
    css=custom_css,
    theme=gr.themes.Soft(primary_hue="purple", secondary_hue="blue"),
    head=head_js
) as demo:

    gr.HTML("<h1 class='main-title'>🇧🇾 TTS Dataset Validator</h1>", sanitize=False)
    gr.HTML("<p class='subtitle'>Аналіз аўдыядатасетаў для выяўлення несупадзенняў паміж метаданымі і гукам</p>", sanitize=False)

    with gr.Row():
        # Left column - Settings
        with gr.Column(scale=1, elem_classes=["settings-panel"]):
            gr.Markdown("### ⚙️ Налады")

            api_key = gr.Textbox(
                label="Gemini API Key",
                type="password",
                value=os.getenv("GOOGLE_API_KEY", ""),
                placeholder="Увядзіце ваш API ключ...",
                elem_id="api_key_input"
            )

            dataset_name = gr.Textbox(
                label="Hugging Face Dataset",
                value="archivartaunik/Melez_Tryvozhnae_Schasce_testpart",
                placeholder="username/dataset_name"
            )

            model_name = gr.Dropdown(
                label="Gemini Мадэль",
                choices=[
                    "gemini-3-pro-preview",
                    "gemini-3-flash-preview",
                    "gemini-2.5-pro",
                    "gemini-2.5-flash",
                    "gemini-2.5-flash-lite",
                    "gemini-2.0-flash",
                    "gemini-2.0-flash-lite",
                ],
                value="gemini-2.5-flash-lite"
            )

            limit_files = gr.Number(
                label="Ліміт файлаў (0 = усе)",
                value=1,
                minimum=0,
                step=1
            )

            gr.Markdown("### 🎛️ Генерацыя")

            temperature = gr.Slider(
                label="Temperature",
                minimum=0.0,
                maximum=2.0,
                value=0.3,
                step=0.1
            )

            thinking_budget = gr.Number(
                label="Thinking Budget (токены)",
                value=1024,
                minimum=0,
                step=128,
                visible=False
            )

            similarity_threshold = gr.Slider(
                label="Парог несупадзенняў (%)",
                minimum=0,
                maximum=100,
                value=99,
                step=1,
                info="Файлы з скорам ніжэй будуць пазначаны"
            )

            analyze_btn = gr.Button(
                "🚀 Пачаць аналіз",
                variant="primary",
                elem_classes=["primary-btn"]
            )

            smart_analyze_btn = gr.Button(
                "🧠 Разумны аналіз",
                variant="secondary",
                elem_classes=["smart-btn"]
            )

            recheck_problematic = gr.Checkbox(
                label="Пераправерыць толькі праблемныя файлы",
                value=False,
                info="Калі ўключана, аналіз будзе запускацца толькі для файлаў з нізкім рэйтынгам."
            )

            stop_btn = gr.Button(
                "🛑 Спыніць аналіз",
                variant="stop",
                elem_classes=["stop-btn"],
                visible=True
            )

            gr.HTML(
                "<small style='color: #94a3b8;'>🧠 Разумны аналіз выкарыстоўвае 3 мадэлі паслядоўна: "
                "Flash-Lite → Flash → Gemini-3-Flash</small>",
                sanitize=False
            )

            gr.Markdown("### � Імпарт CSV")
            with gr.Row():
                import_file = gr.File(label="Загрузіць CSV", file_types=[".csv"])
                import_btn = gr.Button("📥 Імпартаваць", variant="secondary")

            gr.Markdown("### �💾 Кэш")
            cache_status = gr.HTML(value="<p style='color: #94a3b8;'>📭 Кэш пусты</p>", sanitize=False)
            clear_cache_btn = gr.Button("🗑️ Ачысціць кэш датасету", size="sm")

        # Right column - Results
        with gr.Column(scale=2, elem_classes=["results-panel"]):
            gr.Markdown("### 📊 Статыстыка")
            stats_output = gr.HTML(sanitize=False)

            gr.Markdown("### 🚩 Праблемныя файлы")
            # IMPORTANT: sanitize=False so onclick handlers survive
            flagged_output = gr.HTML(sanitize=False)

            with gr.Accordion("📋 Усе вынікі (табліца)", open=False):
                results_table = gr.DataFrame(
                    headers=["path", "score", "model_used", "verification_status", "ref_text", "hyp_text"],
                    wrap=True
                )

            gr.Markdown("### 📤 Экспарт")
            with gr.Row():
                download_btn = gr.Button("💾 Спампаваць вынікі (CSV)", size="lg")
                download_file = gr.File(label="Файл вынікаў", file_count="single")

            # Hidden components for JS->Python communication (rendered, but moved offscreen via CSS)
            verification_data = gr.Textbox(elem_id="verification_data_input", visible=True)
            verification_trigger = gr.Button(elem_id="verification_trigger_btn", visible=True)

    # Event handlers
    model_name.change(
        fn=update_thinking_visibility,
        inputs=[model_name],
        outputs=[thinking_budget]
    )

    analyze_event = analyze_btn.click(
        fn=run_analysis,
        inputs=[api_key, dataset_name, model_name, limit_files, temperature, thinking_budget, similarity_threshold, recheck_problematic],
        outputs=[stats_output, flagged_output, results_table]
    )
    
    analyze_event.then(
        fn=get_cache_status,
        inputs=[],
        outputs=[cache_status]
    )

    smart_analyze_event = smart_analyze_btn.click(
        fn=run_smart_analysis,
        inputs=[api_key, dataset_name, limit_files, temperature, thinking_budget, similarity_threshold, recheck_problematic],
        outputs=[stats_output, flagged_output, results_table]
    )
    
    smart_analyze_event.then(
        fn=get_cache_status,
        inputs=[],
        outputs=[cache_status]
    )

    stop_btn.click(
        fn=None,
        inputs=None,
        outputs=None,
        cancels=[analyze_event, smart_analyze_event]
    )

    download_btn.click(
        fn=save_results_csv,
        inputs=[],
        outputs=[download_file]
    )

    clear_cache_btn.click(
        fn=clear_cache,
        inputs=[],
        outputs=[cache_status]
    )

    import_btn.click(
        fn=import_csv_analysis,
        inputs=[import_file, similarity_threshold, dataset_name, limit_files],
        outputs=[stats_output, flagged_output, results_table]
    )

    # Verification event
    verification_trigger.click(
        fn=verify_action,
        inputs=[verification_data, similarity_threshold],
        outputs=[stats_output, flagged_output, results_table]
    )

if __name__ == "__main__":
    demo.launch(debug=True)
