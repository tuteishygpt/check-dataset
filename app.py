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
    buffer = io.BytesIO()
    sf.write(buffer, audio_array, sampling_rate, format='WAV')
    buffer.seek(0)
    b64_data = base64.b64encode(buffer.read()).decode('utf-8')
    return f'<audio controls src="data:audio/wav;base64,{b64_data}" style="width: 100%; margin-top: 10px; height: 36px; border-radius: 8px;"></audio>'

def generate_dashboard_outputs(similarity_threshold: int):
    """
    Generates the HTML/DF outputs for the dashboard based on global_results.
    Refactored to be used by all analysis functions.
    """
    global global_results
    
    # Create DataFrame
    df = pd.DataFrame(global_results)
    
    if df.empty:
        return "", "", pd.DataFrame()
    
    # Ensure verification_status exists
    if 'verification_status' not in df.columns:
        df['verification_status'] = None
    if 'model_used' not in df.columns:
        df['model_used'] = "unknown"

    # Statistics
    total_files = len(df)
    
    # Problematic are those below threshold AND NOT verified
    # Problematic are those below threshold AND NOT verified as correct
    flagged_mask = (df['score'] < similarity_threshold) & (df['verification_status'] != 'correct')
    
    flagged_count = len(df[flagged_mask])
    avg_score = df['score'].mean() if len(df) > 0 else 0
    
    # Model stats
    model_stats = ""
    if 'model_used' in df.columns:
        model_counts = df['model_used'].value_counts().to_dict()
        model_stats_str = " | ".join([f"{m}: {c}" for m, c in model_counts.items()])
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
            score_color = "#f5576c" if row['score'] < 50 else "#fbbf24" if row['score'] < 75 else "#34d399"
            audio_html = array_to_b64_audio(row['audio_array'], row['sampling_rate'])
            
            flagged_html += f"""
            <div style="background: #1e293b; border-radius: 12px; padding: 20px; margin-bottom: 15px; 
                        border-left: 4px solid {score_color};">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                    <div style="display: flex; flex-direction: column;">
                        <span style="color: #e2e8f0; font-weight: bold;">📄 {row['path']}</span>
                        <div style="margin-top: 5px;">
                             <span style="background: #475569; color: #e2e8f0; padding: 3px 8px; 
                                          border-radius: 10px; font-size: 0.8em; margin-right: 8px;">🤖 {"🖐️ Ручная праверка" if row['model_used'] == 'manual' else row['model_used']}</span>
                        </div>
                    </div>
                    <span style="background: {score_color}; color: white; padding: 5px 12px; 
                                 border-radius: 20px; font-weight: bold;">{row['score']}%</span>
                </div>
                <div style="background: #0f172a; border-radius: 8px; padding: 15px; margin-bottom: 10px;">
                    <p style="color: #94a3b8; margin: 0 0 5px 0; font-size: 0.85em;">📝 Арыгінал:</p>
                    <p style="color: #f1f5f9; margin: 0; font-family: monospace;">{row['ref_text']}</p>
                </div>
                <div style="background: #0f172a; border-radius: 8px; padding: 15px; margin-bottom: 10px;">
                    <p style="color: #94a3b8; margin: 0 0 5px 0; font-size: 0.85em;">🎤 Распазнана:</p>
                    <p style="color: #f1f5f9; margin: 0; font-family: monospace;">{row['hyp_text']}</p>
                </div>
                <details style="color: #94a3b8; margin-bottom: 10px;">
                    <summary style="cursor: pointer; color: #60a5fa;">🔍 Нармалізаваны тэкст</summary>
                    <div style="background: #0f172a; border-radius: 8px; padding: 10px; margin-top: 10px;">
                        <p style="margin: 5px 0;"><strong>Ref:</strong> {row['norm_ref']}</p>
                        <p style="margin: 5px 0;"><strong>Hyp:</strong> {row['norm_hyp']}</p>
                    </div>
                </details>
                {audio_html}
                <div style="display: flex; gap: 10px; margin-top: 15px; justify-content: flex-end;">
                    <button onclick="verifyRecord({row['id']}, 'correct')" class="verify-btn correct-btn">✅ Правільна</button>
                    <button onclick="verifyRecord({row['id']}, 'incorrect')" class="verify-btn incorrect-btn">❌ Няправільна</button>
                </div>
            </div>
            """
    
    # Add minimized rows for manually verified problematic items
    manual_mask = (df['verification_status'].notnull()) & (df['model_used'] == 'manual')
    manual_df = df[manual_mask].sort_values(by="id", ascending=False).head(5)  # Show last 5
    
    if not manual_df.empty:
        flagged_html += """<h4 style="color: #94a3b8; margin: 20px 0 10px 0;">🕒 Апошнія правераныя:</h4>"""
        for _, row in manual_df.iterrows():
            status_icon = "✅" if row['verification_status'] == 'correct' else "❌"
            status_color = "#10b981" if row['verification_status'] == 'correct' else "#ef4444"
            
            flagged_html += f"""
            <div style="background: rgba(30, 41, 59, 0.4); border-radius: 8px; padding: 10px 15px; 
                        margin-bottom: 8px; display: flex; justify-content: space-between; align-items: center;
                        border-left: 3px solid {status_color};">
                <span style="color: #cbd5e1; font-size: 0.9em;">{row['path']}</span>
                <span style="color: {status_color}; font-weight: bold; font-size: 0.9em;">
                    {status_icon} {row['verification_status']} (Score: {row['score']}%)
                </span>
            </div>
            """
    
    # Full table
    # Add verification column if any
    cols = ['path', 'score', 'model_used', 'verification_status', 'ref_text', 'hyp_text']
    
    # Create display DF to avoid modifying global results structure but show nice icons
    display_df = df.copy()
    if 'verification_status' not in display_df.columns:
        display_df['verification_status'] = None
        
    # Map status to symbols for the table
    def map_status(x):
        if x == 'correct': return "✅"
        if x == 'incorrect': return "❌"
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
    progress=gr.Progress()
):
    """
    Main analysis function that processes the dataset.
    """
    global global_results
    
    if not api_key:
        raise gr.Error("Калі ласка, увядзіце Gemini API ключ.")
    
    try:
        # Initialize Client
        client = genai.Client(api_key=api_key)
        
        # Build Config
        config_args = {"temperature": temperature}
        use_thinking = "thinking" in model_name
        
        if use_thinking and thinking_budget > 0:
            config_args["thinking_config"] = {"include_thoughts": True}
        
        gen_config = genai.types.GenerateContentConfig(**config_args)
        
        # Load Dataset (with caching)
        limit = limit_files if limit_files > 0 else None
        
        # Check cache first
        cached_ds = get_cached_dataset(dataset_name, limit)
        if cached_ds is not None:
            progress(0, desc=f"Выкарыстоўваю закэшаваны датасет '{dataset_name}'...")
            ds = cached_ds
        else:
            progress(0, desc=f"Загрузка датасета '{dataset_name}'...")
            ds = utils.load_hf_dataset(dataset_name, limit=limit)
            # Cache the dataset
            cache_dataset(dataset_name, limit, ds)
            progress(0.1, desc=f"Датасет закэшаваны для паўторнага выкарыстання")

        
        results = []
        
        # Process loop
        for idx, item in enumerate(ds):
            progress((idx + 1) / len(ds), desc=f"Апрацоўка файла {idx+1}/{len(ds)}")
            
            audio_data = item['audio']['array']
            sampling_rate = item['audio']['sampling_rate']
            
            # Try common text field names
            ref_text = item.get('sentence') or item.get('text') or item.get('transcription') or item.get('transcript') or ""
            
            # Transcribe
            hyp_text = utils.transcribe_audio(client, model_name, audio_data, sampling_rate, config=gen_config)
            
            # Calculate Score
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
        
        # Store results globally for audio playback
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
    progress=gr.Progress()
):
    """
    Multi-level smart analysis that uses 3 models in sequence.
    
    Steps:
    1. First pass with gemini-2.5-flash-lite on all records
    2. Second pass with gemini-2.5-flash-lite on problematic records only
    3. Third pass with gemini-2.5-flash on remaining problematic records
    4. Fourth pass with gemini-3-flash-preview on remaining problematic records
    """
    global global_results
    
    if not api_key:
        raise gr.Error("Калі ласка, увядзіце Gemini API ключ.")
    
    # Model sequence for smart analysis
    models = [
        ("gemini-2.5-flash-lite", "Этап 1/4: Flash-Lite (першы праход)"),
        ("gemini-2.5-flash-lite", "Этап 2/4: Flash-Lite (другі праход)"),
        ("gemini-2.5-flash", "Этап 3/4: Flash"),
        ("gemini-3-flash-preview", "Этап 4/4: Gemini-3-Flash"),
    ]
    
    try:
        # Initialize Client
        client = genai.Client(api_key=api_key)
        
        # Build Config
        config_args = {"temperature": temperature}
        gen_config = genai.types.GenerateContentConfig(**config_args)
        
        # Load Dataset (with caching)
        limit = limit_files if limit_files > 0 else None
        
        # Check cache first
        cached_ds = get_cached_dataset(dataset_name, limit)
        if cached_ds is not None:
            progress(0, desc=f"Выкарыстоўваю закэшаваны датасет '{dataset_name}'...")
            ds = cached_ds
        else:
            progress(0, desc=f"Загрузка датасета '{dataset_name}'...")
            ds = utils.load_hf_dataset(dataset_name, limit=limit)
            # Cache the dataset
            cache_dataset(dataset_name, limit, ds)
            progress(0.05, desc=f"Датасет закэшаваны для паўторнага выкарыстання")
        
        # Initialize results - first pass processes all records
        results = []
        
        # === STEP 1: First pass with gemini-2.5-flash-lite on ALL records ===
        model_name = models[0][0]
        step_desc = models[0][1]
        progress(0.05, desc=f"{step_desc}: апрацоўка ўсіх {len(ds)} запісаў...")
        
        for idx, item in enumerate(ds):
            progress(
                0.05 + (idx + 1) / len(ds) * 0.20,
                desc=f"{step_desc}: файл {idx+1}/{len(ds)}"
            )
            
            audio_data = item['audio']['array']
            sampling_rate = item['audio']['sampling_rate']
            
            # Try common text field names
            ref_text = item.get('sentence') or item.get('text') or item.get('transcription') or item.get('transcript') or ""
            
            # Transcribe
            hyp_text = utils.transcribe_audio(client, model_name, audio_data, sampling_rate, config=gen_config)
            
            # Calculate Score
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
        
        # === STEP 2-4: Re-analyze problematic records with progressively better models ===
        base_progress = 0.25
        step_progress_size = 0.25  # Each remaining step gets 25% of progress
        
        for step_idx in range(1, len(models)):
            model_name = models[step_idx][0]
            step_desc = models[step_idx][1]
            
            # Find problematic records
            problematic_indices = [
                i for i, r in enumerate(results) 
                if r['score'] < similarity_threshold
            ]
            
            if not problematic_indices:
                progress(
                    base_progress + step_idx * step_progress_size, 
                    desc=f"{step_desc}: няма праблемных запісаў, прапускаем..."
                )
                continue
            
            progress(
                base_progress + (step_idx - 1) * step_progress_size, 
                desc=f"{step_desc}: пераапрацоўка {len(problematic_indices)} праблемных запісаў..."
            )
            
            for j, res_idx in enumerate(problematic_indices):
                progress(
                    base_progress + (step_idx - 1) * step_progress_size + (j + 1) / len(problematic_indices) * step_progress_size,
                    desc=f"{step_desc}: запіс {j+1}/{len(problematic_indices)}"
                )
                
                result = results[res_idx]
                audio_data = result['audio_array']
                sampling_rate = result['sampling_rate']
                ref_text = result['ref_text']
                
                # Re-transcribe with better model
                hyp_text = utils.transcribe_audio(client, model_name, audio_data, sampling_rate, config=gen_config)
                
                # Recalculate Score
                score, norm_ref, norm_hyp = utils.calculate_similarity(ref_text, hyp_text)
                
                # Update result if score improved
                if score > result['score']:
                    # Update status if new score crosses threshold
                    new_status = "correct" if score >= similarity_threshold else "incorrect"
                    results[res_idx].update({
                        "hyp_text": hyp_text,
                        "score": score,
                        "norm_ref": norm_ref,
                        "norm_hyp": norm_hyp,
                        "model_used": model_name,
                        "verification_status": new_status
                    })
        
        # Store results globally for audio playback
        global_results = results
        
        return generate_dashboard_outputs(similarity_threshold)
        
    except Exception as e:
        raise gr.Error(f"Памылка: {e}")


def get_audio_for_row(row_index: int):
    """
    Returns audio for specified row index.
    """
    global global_results
    if row_index < 0 or row_index >= len(global_results):
        return None
    
    row = global_results[row_index]
    buffer = io.BytesIO()
    sf.write(buffer, row['audio_array'], row['sampling_rate'], format='WAV')
    buffer.seek(0)
    return (row['sampling_rate'], np.array(row['audio_array']))


def update_thinking_visibility(model_name: str):
    """
    Show/hide thinking budget based on model selection.
    """
    return gr.update(visible="thinking" in model_name)


def clear_cache():
    """Clear all cached datasets."""
    global dataset_cache
    count = len(dataset_cache)
    dataset_cache.clear()
    return f"<p style='color: #34d399;'>✅ Кэш ачышчаны ({count} датасет(аў) выдалена)</p>"


def get_cache_status():
    """Get current cache status."""
    if not dataset_cache:
        return "<p style='color: #94a3b8;'>📭 Кэш пусты</p>"
    
    items = []
    total_items = 0
    for key in dataset_cache:
        ds_len = len(dataset_cache[key])
        total_items += ds_len
        
    return f"<p style='color: #60a5fa;'>📦 Закэшавана: {len(dataset_cache)} датасет(аў), {total_items} элементаў</p>"



def verify_action(data_str, similarity_threshold):
    """
    Handles verification button click.
    """
    global global_results
    
    try:
        print(f"DEBUG: verify_action called with: {data_str}")
        data = json.loads(data_str)
        record_id = data.get('id')
        status = data.get('status')
        
        if record_id is not None and 0 <= record_id < len(global_results):
            print(f"DEBUG: Updating record {record_id} to {status}")
            # Update the record
            global_results[record_id]['verification_status'] = status
            global_results[record_id]['model_used'] = 'manual'
            
            # Regenerate dashboard
            return generate_dashboard_outputs(similarity_threshold)
            
    except Exception as e:
        print(f"Error in verify_action: {e}")
        return generate_dashboard_outputs(similarity_threshold)
        
    print(f"DEBUG: returning dashboard output")
    
    # Save to CSV to ensure persistence
    try:
        save_df = pd.DataFrame(global_results)
        save_df.to_csv("check_dataset_results.csv", index=False)
        print("Saved results to check_dataset_results.csv")
    except Exception as e:
        print(f"Error saving to CSV: {e}")
        
    return generate_dashboard_outputs(similarity_threshold)



def save_results_csv():
    """
    Saves current results to CSV and returns the path for download.
    """
    global global_results
    if not global_results:
        return None
    
    try:
        df = pd.DataFrame(global_results)
        
        # Exclude heavy columns that don't make sense in CSV
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

.correct-btn {
    background: #10b981;
}

.correct-btn:hover {
    background: #059669;
}

.incorrect-btn {
    background: #ef4444;
}

.incorrect-btn:hover {
    background: #dc2626;
}

footer {
    display: none !important;
}

#verification_data_input, #verification_trigger_btn {
    position: fixed !important;
    top: -10000px !important;
    left: -10000px !important;
    opacity: 0 !important;
    width: 1px !important;
    height: 1px !important;
    pointer-events: auto !important;
    z-index: -1 !important;
}
"""

# JavaScript for localStorage API key persistence
js_localStorage = """
<script>
    // Load API key from localStorage on page load
    document.addEventListener('DOMContentLoaded', function() {
        setTimeout(function() {
            const savedApiKey = localStorage.getItem('gemini_api_key');
            if (savedApiKey) {
                const apiKeyInput = document.querySelector('#api_key_input input');
                if (apiKeyInput && !apiKeyInput.value) {
                    apiKeyInput.value = savedApiKey;
                    apiKeyInput.dispatchEvent(new Event('input', { bubbles: true }));
                }
            }
        }, 500);
    });
    
    // Save API key to localStorage when it changes
    function setupApiKeySaver() {
        const apiKeyInput = document.querySelector('#api_key_input input');
        if (apiKeyInput) {
            apiKeyInput.addEventListener('blur', function() {
                if (this.value && this.value.length > 10) {
                    localStorage.setItem('gemini_api_key', this.value);
                }
            });
        }
    }
    
    // Try to setup the saver after DOM is loaded
    setTimeout(setupApiKeySaver, 1000);

    // Function to trigger verification from HTML buttons
    window.verifyRecord = function(id, status) {
        console.log("verifyRecord v5 calling", id, status);
        const container = document.getElementById('verification_data_input');
        if (!container) {
             console.error("Container not found");
             return;
        }
        
        let textarea = container.querySelector('textarea');
        if (!textarea) textarea = container.querySelector('input');
        
        if (!textarea) {
            console.error("Textarea/Input not found");
            return;
        }
        
        // Add timestamp to ensure change is detected every time
        const newValue = JSON.stringify({id: id, status: status, ts: Date.now()});
        
        // Use native setter to ensure frameworks like React/Svelte detect the change
        const checkTextArea = window.HTMLTextAreaElement.prototype;
        const checkInput = window.HTMLInputElement.prototype;
        
        let setter = Object.getOwnPropertyDescriptor(checkTextArea, "value").set;
        if (!setter) {
             setter = Object.getOwnPropertyDescriptor(checkInput, "value").set;
        }
        
        if (setter) {
            setter.call(textarea, newValue);
        } else {
            textarea.value = newValue;
        }
        
        // Dispatch events to ensure Gradio catches it
        textarea.dispatchEvent(new Event('input', { bubbles: true }));
        
        // Click the trigger button
        setTimeout(() => {
            const triggerBtn = document.getElementById('verification_trigger_btn');
            if (triggerBtn) {
                console.log("Clicking trigger button...");
                triggerBtn.click();
            } else {
                console.error("Trigger button not found");
            }
        }, 200);
    }
</script>
"""

# Build the interface
with gr.Blocks(css=custom_css, theme=gr.themes.Soft(primary_hue="purple", secondary_hue="blue")) as demo:
    
    gr.HTML("<h1 class='main-title'>🇧🇾 TTS Dataset Validator</h1>")
    gr.HTML("<p class='subtitle'>Аналіз аўдыядатасетаў для выяўлення несупадзенняў паміж метаданымі і гукам</p>")
    gr.HTML(js_localStorage)
    
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
                    # Gemini 3 (найнавейшыя)
                    "gemini-3-pro-preview",
                    "gemini-3-flash-preview",
                    # Gemini 2.5
                    "gemini-2.5-pro",
                    "gemini-2.5-flash",
                    "gemini-2.5-flash-lite",
                    # Gemini 2.0
                    "gemini-2.0-flash",
                    "gemini-2.0-flash-lite",
                ],
                value="gemini-2.5-flash-lite"
            )
            
            limit_files = gr.Number(
                label="Ліміт файлаў (0 = усе)",
                value=10,
                minimum=0,
                step=1
            )
            
            gr.Markdown("### 🎛️ Генерацыя")
            
            temperature = gr.Slider(
                label="Temperature",
                minimum=0.0,
                maximum=2.0,
                value=1.0,
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
                value=90,
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
            gr.Markdown(
                "<small style='color: #94a3b8;'>🧠 Разумны аналіз выкарыстоўвае 3 мадэлі паслядоўна: "
                "Flash-Lite → Flash → Gemini-3-Flash</small>"
            )
            
            gr.Markdown("### 💾 Кэш")
            cache_status = gr.HTML(
                value="<p style='color: #94a3b8;'>📭 Кэш пусты</p>"
            )
            clear_cache_btn = gr.Button(
                "🗑️ Ачысціць кэш датасету",
                size="sm"
            )
        
        # Right column - Results
        with gr.Column(scale=2, elem_classes=["results-panel"]):
            gr.Markdown("### 📊 Статыстыка")
            stats_output = gr.HTML()
            
            gr.Markdown("### 🚩 Праблемныя файлы")
            flagged_output = gr.HTML()
            
            with gr.Accordion("📋 Усе вынікі (табліца)", open=False):
                results_table = gr.DataFrame(
                    headers=["path", "score", "model_used", "verification_status", "ref_text", "hyp_text"],
                    wrap=True
                )
            
            gr.Markdown("### 📤 Экспарт")
            with gr.Row():
                download_btn = gr.Button("💾 Спампаваць вынікі (CSV)", size="lg")
                download_file = gr.File(label="Файл вынікаў", file_count="single")
            
            # Hidden components for JS->Python communication (visible=True but hidden via CSS)
            verification_data = gr.Textbox(elem_id="verification_data_input", visible=True)
            verification_trigger = gr.Button(elem_id="verification_trigger_btn", visible=True)
            

    
    # Event handlers
    model_name.change(
        fn=update_thinking_visibility,
        inputs=[model_name],
        outputs=[thinking_budget]
    )
    
    analyze_btn.click(
        fn=run_analysis,
        inputs=[
            api_key,
            dataset_name,
            model_name,
            limit_files,
            temperature,
            thinking_budget,
            similarity_threshold
        ],
        outputs=[stats_output, flagged_output, results_table]
    ).then(
        fn=get_cache_status,
        inputs=[],
        outputs=[cache_status]
    )
    
    smart_analyze_btn.click(
        fn=run_smart_analysis,
        inputs=[
            api_key,
            dataset_name,
            limit_files,
            temperature,
            thinking_budget,
            similarity_threshold
        ],
        outputs=[stats_output, flagged_output, results_table]
    ).then(
        fn=get_cache_status,
        inputs=[],
        outputs=[cache_status]
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
    
    # Verification event
    verification_trigger.click(
        fn=verify_action,
        inputs=[verification_data, similarity_threshold],
        outputs=[stats_output, flagged_output, results_table]
    )


if __name__ == "__main__":
    demo.launch(debug=True)

