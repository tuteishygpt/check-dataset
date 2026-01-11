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
import re
from datasets import Dataset, Audio, Features, Value
from huggingface_hub import login, HfApi

from gemini_api import GeminiIntegrator, BatchTask
import tempfile
import shutil


# Load environment variables
load_dotenv()

# Global variable to store results for audio playback
global_results = []

# Cache for downloaded datasets
dataset_cache = {}

def sanitize_filename(name):
    """Sanitize string to be used as a filename."""
    if not name:
        return "results"
    # Replace non-alphanumeric with undersore
    s = re.sub(r'[^\w\s-]', '_', name).strip().lower()
    # Replace whitespace with underscore
    s = re.sub(r'[-\s]+', '_', s)
    return s


def select_best_model_result(model_results: dict, similarity_threshold: int = 90):
    """
    Параўноўвае вынікі ўсіх мадэлей і выбірае лепшы вынік.
    
    Args:
        model_results: Слоўнік {model_name: {"hyp_text": ..., "score": ..., "norm_ref": ..., "norm_hyp": ...}}
        similarity_threshold: Парог для вызначэння карэктнасці
    
    Returns:
        Tuple (best_model_name, best_result_dict)
    """
    if not model_results:
        return None, None
    
    best_model = None
    best_result = None
    best_score = -1
    
    for model_name, result in model_results.items():
        score = result.get('score', 0)
        if score > best_score:
            best_score = score
            best_model = model_name
            best_result = result
    
    return best_model, best_result


def compare_two_models(model_results: dict, model1: str, model2: str):
    """
    Параўноўвае вынікі двух канкрэтных мадэлей.
    
    Args:
        model_results: Слоўнік з вынікамі ўсіх мадэлей
        model1: Назва першай мадэлі
        model2: Назва другой мадэлі
    
    Returns:
        Слоўнік з параўнаннем: {"model1": ..., "model2": ..., "winner": ..., "score_diff": ...}
    """
    result1 = model_results.get(model1)
    result2 = model_results.get(model2)
    
    if not result1 and not result2:
        return {"error": "Абедзве мадэлі не знойдзены"}
    if not result1:
        return {"winner": model2, "model1": None, "model2": result2, "score_diff": None}
    if not result2:
        return {"winner": model1, "model1": result1, "model2": None, "score_diff": None}
    
    score1 = result1.get('score', 0)
    score2 = result2.get('score', 0)
    score_diff = abs(score1 - score2)
    
    if score1 > score2:
        winner = model1
    elif score2 > score1:
        winner = model2
    else:
        winner = "tie"
    
    return {
        "model1": {"name": model1, **result1},
        "model2": {"name": model2, **result2},
        "winner": winner,
        "score_diff": score_diff
    }


def get_all_model_comparison(record: dict):
    """
    Атрымлівае поўнае параўнанне вынікаў усіх мадэлей для запісу.
    
    Args:
        record: Запіс з global_results
    
    Returns:
        Слоўнік з усімі параўнаннямі і лепшым вынікам
    """
    model_results = record.get('model_results', {})
    
    if not model_results:
        return {
            "models_count": 0,
            "best_model": None,
            "comparisons": [],
            "all_scores": {}
        }
    
    best_model, best_result = select_best_model_result(model_results)
    
    # Стварыць усе парныя параўнанні
    models = list(model_results.keys())
    comparisons = []
    
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            comparison = compare_two_models(model_results, models[i], models[j])
            comparisons.append(comparison)
    
    # Сабраць усе скоры
    all_scores = {model: result.get('score', 0) for model, result in model_results.items()}
    
    return {
        "models_count": len(models),
        "best_model": best_model,
        "best_result": best_result,
        "comparisons": comparisons,
        "all_scores": all_scores
    }


def find_best_model_pair(record: dict, ref_text: str):
    """
    Знаходзіць пару крыніц з найлепшым супадзеннем ПАМІЖ САБОЙ.
    
    Args:
        record: Запіс з global_results
        ref_text: Арыгінальны тэкст для параўнання
    
    Returns:
        Слоўнік з інфармацыяй пра лепшую пару
    """
    model_results = record.get('model_results', {})
    
    if not model_results or len(model_results) < 2:
        return None
    
    # Параўнаць усе пары крыніц паміж сабой і знайсці найлепшае супадзенне
    sources = list(model_results.items())
    best_pair = None
    best_pair_similarity = -1
    
    import utils
    
    for i in range(len(sources)):
        for j in range(i + 1, len(sources)):
            m1_name, m1_result = sources[i]
            m2_name, m2_result = sources[j]
            
            m1_hyp = m1_result.get('hyp_text', '')
            m2_hyp = m2_result.get('hyp_text', '')
            
            # Вылічыць падабенства паміж двума гіпотэзамі
            pair_similarity, _, _ = utils.calculate_similarity(m1_hyp, m2_hyp)
            
            if pair_similarity > best_pair_similarity:
                best_pair_similarity = pair_similarity
                m1_ref_score = m1_result.get('score', 0)
                m2_ref_score = m2_result.get('score', 0)
                
                # Выбраць лепшы тэкст (той, што мае вышэйшы скор з арыгіналам)
                if m1_ref_score >= m2_ref_score:
                    best_hyp = m1_hyp
                    best_model = m1_name
                    best_score = m1_ref_score
                else:
                    best_hyp = m2_hyp
                    best_model = m2_name
                    best_score = m2_ref_score
                
                best_pair = {
                    "model1": m1_name,
                    "model2": m2_name,
                    "model1_hyp": m1_hyp,
                    "model2_hyp": m2_hyp,
                    "model1_score": m1_ref_score,
                    "model2_score": m2_ref_score,
                    "pair_similarity": pair_similarity,
                    "best_hyp": best_hyp,
                    "best_model": best_model,
                    "best_score": best_score
                }
    
    return best_pair

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
        # LIMIT displayed bad files to prevent Browser Crash (OOM)
        max_display = 50
        
        if len(flagged_df) > max_display:
            flagged_html += f"""
            <div style="background: #f59e0b; padding: 15px; border-radius: 8px; margin-bottom: 20px; color: white; text-align: center;">
                ⚠️ Паказана першыя {max_display} з {len(flagged_df)} праблемных файлаў, каб пазбегнуць перагрузкі браўзера.
            </div>
            """
        
        for i, (_, row) in enumerate(flagged_df.iterrows()):
            if i >= max_display:
                break
                
            rid = int(row['id']) if pd.notnull(row.get('id')) else -1
            score = float(row['score']) if pd.notnull(row.get('score')) else 0.0
            score_int = int(round(score))
            score_color = "#f5576c" if score < 50 else "#fbbf24" if score < 75 else "#34d399"

            audio_html = array_to_b64_audio(row.get('audio_array'), row.get('sampling_rate'))

            model_used = row.get('model_used', 'unknown')
            model_badge = "🖐️ Ручная праверка" if model_used == 'manual' else _e(model_used)
            
            # Генерацыя HTML для параўнання мадэлей
            model_comparison_html = ""
            model_results = row.get('model_results', {})
            if model_results and len(model_results) > 1:
                comparison_data = get_all_model_comparison(row)
                best_model = comparison_data.get('best_model', '')
                
                model_scores_rows = ""
                # Сартаваць па скору
                sorted_models = sorted(model_results.items(), key=lambda x: -x[1].get('score', 0))
                
                for m_name, m_data in sorted_models:
                    m_score = m_data.get('score', 0)
                    m_hyp = m_data.get('hyp_text', '')
                    is_best = "✅" if m_name == best_model else ""
                    score_bg = "#10b981" if m_score >= similarity_threshold else "#f59e0b" if m_score >= 70 else "#ef4444"
                    
                    model_scores_rows += f"""
                        <tr style="border-bottom: 1px solid #334155;">
                            <td style="color: #e2e8f0; padding: 8px 10px; vertical-align: top; white-space: nowrap;">
                                <div style="font-weight: bold; margin-bottom: 4px;">{is_best} {_e(m_name)}</div>
                                <span style="background: {score_bg}; color: white; padding: 2px 8px; border-radius: 10px; font-size: 0.8em;">{int(m_score)}%</span>
                            </td>
                            <td style="color: #cbd5e1; padding: 8px 10px; font-family: monospace; font-size: 0.85em; vertical-align: top;">
                                {_e(m_hyp)}
                            </td>
                        </tr>
                    """
                
                model_comparison_html = f"""
                <details style="color: #94a3b8; margin-bottom: 15px;">
                    <summary style="cursor: pointer; color: #60a5fa; margin-bottom: 5px;">📊 Параўнанне мадэлей ({len(model_results)})</summary>
                    <div style="background: #0f172a; border-radius: 8px; overflow: hidden; border: 1px solid #334155; margin-top: 5px;">
                        <table style="width: 100%; border-collapse: collapse;">
                            <thead>
                                <tr style="background: #1e293b; border-bottom: 1px solid #475569;">
                                    <th style="text-align: left; color: #94a3b8; padding: 10px; width: 30%;">Мадэль / Скор</th>
                                    <th style="text-align: left; color: #94a3b8; padding: 10px;">Тэкст</th>
                                </tr>
                            </thead>
                            <tbody>
                                {model_scores_rows}
                            </tbody>
                        </table>
                    </div>
                </details>
                """

            # Атрымаць лепшы вынік (пара крыніц з найлепшым супадзеннем паміж сабой)
            best_text_html = ""
            ref_text = row.get('ref_text', '')
            
            if model_results and len(model_results) >= 2:
                # Выклікаем функцыю, якая цяпер шукае лепшае супадзенне ПАМІЖ КРЫНІЦАМІ
                best_pair = find_best_model_pair(row, ref_text)
                if best_pair:
                    m1_name = best_pair.get('model1', '')
                    m2_name = best_pair.get('model2', '')
                    pair_sim = best_pair.get('pair_similarity', 0)
                    best_hyp_pair = best_pair.get('best_hyp', '')
                    
                    pair_sim_bg = "#10b981" if pair_sim >= 95 else "#f59e0b" if pair_sim >= 80 else "#ef4444"
                    
                    best_text_html = f"""
                <div style="background: linear-gradient(135deg, #1e3a5f 0%, #0f172a 100%); border-radius: 8px; padding: 15px; margin-bottom: 10px; border: 1px solid #3b82f6;">
                    <p style="color: #60a5fa; margin: 0 0 10px 0; font-size: 0.9em;">
                        🏆 Найлепшае супадзенне паміж крыніцамі:
                        <span style="background: #475569; color: #e2e8f0; padding: 2px 8px; border-radius: 6px; margin: 0 5px;">{_e(m1_name)}</span>
                        ↔
                        <span style="background: #475569; color: #e2e8f0; padding: 2px 8px; border-radius: 6px; margin: 0 5px;">{_e(m2_name)}</span>
                        <span style="background: {pair_sim_bg}; color: white; padding: 2px 8px; border-radius: 10px; font-weight: bold;">{int(pair_sim)}%</span>
                    </p>
                    <p style="color: #93c5fd; margin: 0; font-family: monospace; font-weight: bold;">{_e(best_hyp_pair)}</p>
                    <button type="button" onclick="verifyRecord({rid}, 'update_match')" class="verify-btn" style="background: #3b82f6; margin-top: 10px; width: 100%; font-size: 0.9em;">📝 Замяніць арыгінал і пацвердзіць</button>
                </div>
                """
            elif model_results and len(model_results) == 1:
                # Толькі адна мадэль
                m_name = list(model_results.keys())[0]
                res = model_results[m_name]
                score = res.get('score', 0)
                hyp = res.get('hyp_text', '')
                score_bg = "#10b981" if score >= similarity_threshold else "#f59e0b" if score >= 70 else "#ef4444"
                
                best_text_html = f"""
                <div style="background: linear-gradient(135deg, #1e3a5f 0%, #0f172a 100%); border-radius: 8px; padding: 15px; margin-bottom: 10px; border: 1px solid #3b82f6;">
                    <p style="color: #60a5fa; margin: 0 0 5px 0; font-size: 0.9em;">
                        🏆 Вынік ({_e(m_name)}) 
                        <span style="background: {score_bg}; color: white; padding: 2px 8px; border-radius: 10px; margin-left: 8px;">{int(score)}%</span>
                    </p>
                    <p style="color: #93c5fd; margin: 0; font-family: monospace; font-weight: bold;">{_e(hyp)}</p>
                    <button type="button" onclick="verifyRecord({rid}, 'update_match')" class="verify-btn" style="background: #3b82f6; margin-top: 10px; width: 100%; font-size: 0.9em;">📝 Замяніць арыгінал і пацвердзіць</button>
                </div>
                """
            
            # Атрымаць скор і мадэль для бягучага "Распазнана"
            current_hyp = row.get('hyp_text', '')
            current_model = row.get('model_used', 'unknown')
            current_score = float(row.get('score', 0))
            current_score_bg = "#10b981" if current_score >= similarity_threshold else "#f59e0b" if current_score >= 70 else "#ef4444"

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
                    <p style="color: #94a3b8; margin: 0 0 5px 0; font-size: 0.85em;">
                        🎤 Распазнана 
                        <span style="background: #475569; color: #e2e8f0; padding: 2px 6px; border-radius: 6px; font-size: 0.9em; margin-left: 5px;">{_e(current_model)}</span>
                        <span style="background: {current_score_bg}; color: white; padding: 2px 6px; border-radius: 6px; font-size: 0.9em; margin-left: 5px;">{int(current_score)}%</span>
                    </p>
                    <p style="color: #f1f5f9; margin: 0; font-family: monospace;">{_e(current_hyp)}</p>
                </div>
                {best_text_html}
                <details style="color: #94a3b8; margin-bottom: 10px;">
                    <summary style="cursor: pointer; color: #60a5fa;">🔍 Нармалізаваны тэкст</summary>
                    <div style="background: #0f172a; border-radius: 8px; padding: 10px; margin-top: 10px;">
                        <p style="margin: 5px 0;"><strong>Ref:</strong> {_e(row.get('norm_ref'))}</p>
                        <p style="margin: 5px 0;"><strong>Hyp:</strong> {_e(row.get('norm_hyp'))}</p>
                    </div>
                </details>
                {model_comparison_html}
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
    batch_mode: bool = False,
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
                    ds = utils.load_hf_dataset(dataset_name, limit=limit)
                    cache_dataset(dataset_name, limit, ds)
                
                # Init results if fresh run
                progress(0.1, desc="Ініцыялізацыя спісу...")
                global_results = []
                for idx, item in enumerate(ds):
                    ref_text = item.get('sentence') or item.get('text') or item.get('transcription') or item.get('transcript') or ""
                    global_results.append({
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
            task_map_idx = {} # task_key -> result_index
            
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
                    ds_map[di] = d_item # Also index by integer ID if needed

                for global_res_idx in target_indices:
                    res = global_results[global_res_idx]
                    path = res.get('path', '')
                    
                    # Try finding item
                    item = ds_map.get(path) or ds_map.get(os.path.basename(path))
                    if not item and res.get('id') is not None:
                         # Try by ID if it's an integer
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
                    # res corresponds to ds[idx]
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
            prompt = "Transcribe the following audio verbatim in Belarusian."
            
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

        # ---------------------------------------------------------
        # STANDARD SYNC MODE
        # ---------------------------------------------------------
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
                        # Also update the global_results with audio for future use
                        global_results[idx]['audio_array'] = audio_data
                        global_results[idx]['sampling_rate'] = sampling_rate
                    else:
                        print(f"Propblematic Recheck: Skipping index {idx}, path '{path}', id {result.get('id')}: Audio not found in dataset.")
                        continue  # Skip if audio still not found

                hyp_text = gemini_tool.transcribe_audio(model_name, audio_data, sampling_rate, config=gen_config)
                score, norm_ref, norm_hyp = utils.calculate_similarity(ref_text, hyp_text)

                print(f"🔄 Updated: {result.get('path')} | Score: {result.get('score')} -> {score} | Text: {hyp_text}")

                # Захаваць вынік гэтай мадэлі ў model_results
                if 'model_results' not in global_results[idx]:
                    global_results[idx]['model_results'] = {}
                
                global_results[idx]['model_results'][model_name] = {
                    "hyp_text": hyp_text,
                    "score": score,
                    "norm_ref": norm_ref,
                    "norm_hyp": norm_hyp
                }
                
                # Параўнаць і выбраць лепшы вынік з усіх мадэлей
                best_model, best_result = select_best_model_result(
                    global_results[idx]['model_results'], 
                    similarity_threshold
                )
                
                # Абнавіць асноўныя палі лепшым вынікам
                if best_result:
                    global_results[idx].update({
                        "hyp_text": best_result['hyp_text'],
                        "score": best_result['score'],
                        "norm_ref": best_result['norm_ref'],
                        "norm_hyp": best_result['norm_hyp'],
                        "model_used": best_model,
                        "verification_status": "correct" if best_result['score'] >= similarity_threshold else "incorrect"
                    })

        else:
            # Normal sync run
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
            
            global_results = results

        return generate_dashboard_outputs(similarity_threshold)

    except Exception as e:
        raise gr.Error(f"Памылка: {e}")

    ''' Garbage starts
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
                    ds = utils.load_hf_dataset(dataset_name, limit=limit)
                    cache_dataset(dataset_name, limit, ds)
                    
                # Init results if fresh run
                global_results = [] # Reset global_results for a fresh run

            # 2. PROCESSING in batch mode
            tasks = []
            tmp_dir_obj = tempfile.TemporaryDirectory() # Keep ref to avoid deletion until function ends
            tmp_dir = tmp_dir_obj.name
            
            # Helper to get task from dataset item or result record
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

            progress(0.1, desc="Падрыхтоўка задач для пакетнага рэжыму...")
            
            task_map_idx = {} # task_key -> result_index in global_results
            
            if recheck_problematic:
                # Build tasks for recheck
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

                # Create a map for quick dataset item lookup
                ds_map = {}
                for di, d_item in enumerate(ds):
                    ds_map[di] = d_item
                    p = d_item['audio']['path']
                    if p:
                        ds_map[p] = d_item
                        ds_map[os.path.basename(p)] = d_item

                for i, global_res_idx in enumerate(target_indices):
                    res = global_results[global_res_idx]
                    item = None
                    
                    path = res.get('path', '')
                    item = ds_map.get(path) or ds_map.get(os.path.basename(path))
                    if not item and res.get('id') is not None:
                        item = ds_map.get(int(res.get('id')))
                    
                    if item:
                        t = prepare_task(global_res_idx, res, item)
                        if t:
                            tasks.append(t)
                            task_map_idx[t.key] = global_res_idx
            else:
                # Fresh run
                for idx, item in enumerate(ds):
                    # Init placeholder result in global_results
                    ref_text = item.get('sentence') or item.get('text') or item.get('transcription') or item.get('transcript') or ""
                    
                    global_results.append({
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
                    
                    t = prepare_task(idx, global_results[-1], item)
                    if t:
                        tasks.append(t)
                        task_map_idx[t.key] = idx

            if not tasks:
                gr.Warning("Не знойдзена задач для выканання (магчыма, адсутнічае аўдыя).")
                return generate_dashboard_outputs(similarity_threshold)

            # EXECUTE BATCH
            progress(0.3, desc=f"Запуск пакетнай апрацоўкі ({len(tasks)} файлаў)...")
            prompt = "Transcribe the following audio verbatim in Belarusian."
            batch_results = gemini_tool.run_batch(tasks, model_name, prompt)
            
            progress(0.9, desc="Апрацоўка вынікаў...")
            
            # Map results back
            for key, text in batch_results.items():
                if key in task_map_idx:
                    idx = task_map_idx[key]
                    if idx < len(global_results):
                        # Transcribe done, calc metrics
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
                        
                        # Also populate model_results
                        if 'model_results' not in global_results[idx]:
                            global_results[idx]['model_results'] = {}
                        global_results[idx]['model_results'][model_name] = {
                            "hyp_text": text,
                            "score": score,
                            "norm_ref": norm_ref,
                            "norm_hyp": norm_hyp
                        }

            try:
                tmp_dir_obj.cleanup()
            except: pass
            
            return generate_dashboard_outputs(similarity_threshold)

        else: # STANDARD SYNC MODE (Legacy Logic)
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
                            # Also update the global_results with audio for future use
                            global_results[idx]['audio_array'] = audio_data
                            global_results[idx]['sampling_rate'] = sampling_rate
                        else:
                            print(f"Propblematic Recheck: Skipping index {idx}, path '{path}', id {result.get('id')}: Audio not found in dataset.")
                            continue  # Skip if audio still not found

                    hyp_text = gemini_tool.transcribe_audio(model_name, audio_data, sampling_rate, config=gen_config)
                    score, norm_ref, norm_hyp = utils.calculate_similarity(ref_text, hyp_text)

                    print(f"🔄 Updated: {result.get('path')} | Score: {result.get('score')} -> {score} | Text: {hyp_text}")

                    # Захаваць вынік гэтай мадэлі ў model_results
                    if 'model_results' not in global_results[idx]:
                        global_results[idx]['model_results'] = {}
                    
                    global_results[idx]['model_results'][model_name] = {
                        "hyp_text": hyp_text,
                        "score": score,
                        "norm_ref": norm_ref,
                        "norm_hyp": norm_hyp
                    }
                    
                    # Параўнаць і выбраць лепшы вынік з усіх мадэлей
                    best_model, best_result = select_best_model_result(
                        global_results[idx]['model_results'], 
                        similarity_threshold
                    )
                    
                    # Абнавіць асноўныя палі лепшым вынікам
                    if best_result:
                        global_results[idx].update({
                            "hyp_text": best_result['hyp_text'],
                            "score": best_result['score'],
                            "norm_ref": best_result['norm_ref'],
                            "norm_hyp": best_result['norm_hyp'],
                            "model_used": best_model,
                            "verification_status": "correct" if best_result['score'] >= similarity_threshold else "incorrect"
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
                
                global_results = results

            return generate_dashboard_outputs(similarity_threshold)

    except Exception as e:
        raise gr.Error(f"Памылка: {e}")
    '''


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
        gemini_tool = GeminiIntegrator(api_key=api_key)

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
                        # Also update with audio for future use
                        results[res_idx]['audio_array'] = audio_data
                        results[res_idx]['sampling_rate'] = sampling_rate
                    else:
                         print(f"Smart Analysis Recheck: Skipping index {res_idx}, path '{path}', id {result.get('id')}: Audio not found.")
                         continue  # Skip if audio still not found

                hyp_text = gemini_tool.transcribe_audio(model_name, audio_data, sampling_rate, config=gen_config)
                score, norm_ref, norm_hyp = utils.calculate_similarity(ref_text, hyp_text)

                print(f"🔄 Smart Updated (Step 1): {result.get('path')} | Score: {result.get('score')} -> {score} | Text: {hyp_text}")

                # Захаваць вынік гэтай мадэлі ў model_results
                if 'model_results' not in results[res_idx]:
                    results[res_idx]['model_results'] = {}
                
                results[res_idx]['model_results'][model_name] = {
                    "hyp_text": hyp_text,
                    "score": score,
                    "norm_ref": norm_ref,
                    "norm_hyp": norm_hyp
                }
                
                # Параўнаць і выбраць лепшы вынік з усіх мадэлей
                best_model, best_result = select_best_model_result(
                    results[res_idx]['model_results'], 
                    similarity_threshold
                )
                
                # Абнавіць асноўныя палі лепшым вынікам
                if best_result:
                    results[res_idx].update({
                        "hyp_text": best_result['hyp_text'],
                        "score": best_result['score'],
                        "norm_ref": best_result['norm_ref'],
                        "norm_hyp": best_result['norm_hyp'],
                        "model_used": best_model,
                        "verification_status": "correct" if best_result['score'] >= similarity_threshold else "incorrect"
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

                hyp_text = gemini_tool.transcribe_audio(model_name, audio_data, sampling_rate, config=gen_config)
                score, norm_ref, norm_hyp = utils.calculate_similarity(ref_text, hyp_text)

                # Заўсёды захоўваць вынік гэтай мадэлі ў model_results
                if 'model_results' not in results[res_idx]:
                    results[res_idx]['model_results'] = {}
                
                results[res_idx]['model_results'][model_name] = {
                    "hyp_text": hyp_text,
                    "score": score,
                    "norm_ref": norm_ref,
                    "norm_hyp": norm_hyp
                }
                
                # Параўнаць і выбраць лепшы вынік з усіх мадэлей
                best_model, best_result = select_best_model_result(
                    results[res_idx]['model_results'], 
                    similarity_threshold
                )
                
                # Абнавіць асноўныя палі лепшым вынікам
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
        
        # Deduplicate based on file_name or path if present
        # Prefer 'file_name' then 'path'
        dedup_col = None
        if 'file_name' in df.columns:
            dedup_col = 'file_name'
        elif 'path' in df.columns:
            dedup_col = 'path'
            
        if dedup_col:
            initial_len = len(df)
            df = df.drop_duplicates(subset=[dedup_col])
            if len(df) < initial_len:
                print(f"Import CSV: Removed {initial_len - len(df)} duplicate rows based on '{dedup_col}'.")
        
        # Standardize column names based on known formats
        # Priority 1: Exported format columns: id, path, ref_text, hyp_text, score, norm_ref, norm_hyp, model_used, verification_status
        # Priority 2: Simple/Ad-hoc format: idx, file_name, score_%, ref, hyp
        
        rename_map = {}
        # Mapping definition
        if 'file_name' in df.columns and 'path' not in df.columns:
            rename_map['file_name'] = 'path'
        if 'idx' in df.columns and 'id' not in df.columns:
            rename_map['idx'] = 'id'
        if 'score_%' in df.columns and 'score' not in df.columns:
             rename_map['score_%'] = 'score'
        if 'ref' in df.columns and 'ref_text' not in df.columns:
             rename_map['ref'] = 'ref_text'
        if 'hyp' in df.columns and 'hyp_text' not in df.columns:
             rename_map['hyp'] = 'hyp_text'
             
        if rename_map:
            df = df.rename(columns=rename_map)
        
        # Try to load audio from dataset cache if available
        audio_map = {}
        limit = None # Ignore limit when importing to find all matching audio
        cached_ds = get_cached_dataset(dataset_name, limit)

        if not cached_ds:
             try:
                 print(f"Importing CSV: Loading dataset '{dataset_name}' to link audio (filtered)...")
                 
                 # Collect target filenames to avoid loading full dataset and crashing memory
                 target_paths = set()
                 for _, r_row in df.iterrows():
                     # Use 'path' column as primary source for filename
                     fname_t = str(r_row.get('path', ''))
                     if fname_t:
                         target_paths.add(fname_t)
                         target_paths.add(os.path.basename(fname_t))
                 
                 ds = utils.load_hf_dataset(dataset_name, limit=limit, allowed_paths=target_paths)
                 # We simply use this partial dataset to populate audio_map. 
                 # We do NOT cache it globally as 'full' dataset because it's partial.
                 cached_ds = ds 
             except Exception as e:
                 print(f"Warning: Could not load dataset '{dataset_name}' during import: {e}")
                 cached_ds = []
        
        if cached_ds:
             for item in cached_ds:
                 path = item['audio']['path']
                 if path:
                     fname = os.path.basename(path)
                     audio_map[fname] = item
                     audio_map[path] = item # Store both full path and basename
        else:
             # Load dataset if not cached to ensure audio is available (fallback)
             try:
                 print(f"Importing CSV: Loading dataset '{dataset_name}' to link audio...")
                 ds = utils.load_hf_dataset(dataset_name, limit=limit)
                 cache_dataset(dataset_name, limit, ds)
                 for item in ds:
                     path = item['audio']['path']
                     if path:
                         fname = os.path.basename(path)
                         audio_map[fname] = item
                         audio_map[path] = item
             except Exception as e:
                 print(f"Warning: Could not load dataset '{dataset_name}' during import: {e}")

        results = []
        
        # Helper to safely get string from potential NaN
        def safe_str(val, default=''):
            if pd.isna(val): return default
            return str(val)

        for idx, row in df.iterrows():
            # Get values using standard column names (after rename)
            fname = safe_str(row.get('path', ''))
            ref = safe_str(row.get('ref_text', ''))
            hyp = safe_str(row.get('hyp_text', ''))
            norm_ref = safe_str(row.get('norm_ref', ''))
            norm_hyp = safe_str(row.get('norm_hyp', ''))

            # Calculate similarity and normalization using the standard function
            score, norm_ref, norm_hyp = utils.calculate_similarity(ref, hyp)

            model_used_val = row.get('model_used')
            # If model_used is nan, decide default: 'imported_csv'
            model_used = safe_str(model_used_val, 'imported_csv')
            
            verification_status_val = row.get('verification_status')
            # If status is nan, decide default: 'unknown'
            verification_status = safe_str(verification_status_val, 'unknown')
            
            # Apply automatic verification based on threshold
            # "калі гэты тэкст праходзіць па зададзенаму 'Парог несупадзенняў (%)' то адзначаць, што запіс карэктны"
            # Use rounded score to match UI display (e.g. 98.6% -> 99% should pass 99% threshold)
            if int(round(score)) >= similarity_threshold:
                verification_status = 'correct'
            
            row_id = row.get('id', idx)

            # Find audio
            audio_array = None
            sampling_rate = None
            
            # Try exact match or basename match
            item = audio_map.get(fname)
            if not item:
                 item = audio_map.get(os.path.basename(fname))
            
            if item:
                audio_array = item['audio']['array']
                sampling_rate = item['audio']['sampling_rate']
            
            # Append result with all metadata
            # Загрузіць model_results з JSON калі ёсць
            model_results = {}
            model_results_val = row.get('model_results')
            if pd.notnull(model_results_val) and model_results_val:
                try:
                    model_results = json.loads(str(model_results_val))
                except:
                    model_results = {}
            
            # Дадаць імпартаваны вынік як крыніцу ў model_results
            if hyp and score > 0:
                source_name = f"imported_{model_used}" if model_used != 'imported_csv' else 'imported_csv'
                model_results[source_name] = {
                    "hyp_text": hyp,
                    "score": score,
                    "norm_ref": norm_ref,
                    "norm_hyp": norm_hyp
                }
            
            results.append({
                "id": int(row_id) if pd.notnull(row_id) else idx,
                "path": fname,
                "score": score,
                "ref_text": ref,
                "hyp_text": hyp,
                "audio_array": audio_array,
                "sampling_rate": sampling_rate,
                "status": "processed",
                # Restore Verification Status and Model Used
                "verification_status": verification_status,
                "model_used": model_used,
                "norm_ref": norm_ref,
                "norm_hyp": norm_hyp,
                "model_results": model_results
            })
            
        global_results = results
        print(f"Imported {len(results)} results from CSV.")
        return generate_dashboard_outputs(similarity_threshold)
        
    except Exception as e:
        print(f"Error importing CSV: {e}")
        # Return empty outputs on error
        return "", "", pd.DataFrame()



def _find_index_by_id(record_id: int):
    """Find index in global_results by record['id'] (not by list position)."""
    global global_results
    for i, r in enumerate(global_results):
        if r.get("id") == record_id:
            return i
    return None


def verify_action(data_str, similarity_threshold, dataset_name):
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

        if record_id is None:
            return generate_dashboard_outputs(similarity_threshold)

        if status not in ("correct", "incorrect", "update_match"):
            return generate_dashboard_outputs(similarity_threshold)

        idx = _find_index_by_id(int(record_id))
        if idx is None:
            return generate_dashboard_outputs(similarity_threshold)

        if status == 'update_match':
            # Logic to find best text and update reference
            record = global_results[idx]
            model_results = record.get('model_results', {})
            ref_text = record.get('ref_text', '')
            best_text = ""
            
            if model_results:
                if len(model_results) >= 2:
                    best_pair = find_best_model_pair(record, ref_text)
                    if best_pair:
                        best_text = best_pair.get('best_hyp', '')
                    else:
                         best_model, best_res = select_best_model_result(model_results)
                         if best_res:
                            best_text = best_res.get('hyp_text', '')
                else:
                    best_model, best_res = select_best_model_result(model_results)
                    if best_res:
                        best_text = best_res.get('hyp_text', '')
            
            if best_text:
                global_results[idx]['ref_text'] = best_text
                global_results[idx]['verification_status'] = 'correct'
                global_results[idx]['model_used'] = 'manual'
                
                # Recalculate scores against new reference
                for m_name, m_res in model_results.items():
                    hyp = m_res.get('hyp_text', '')
                    new_score, _, _ = utils.calculate_similarity(best_text, hyp)
                    global_results[idx]['model_results'][m_name]['score'] = new_score
                
                # Update main score
                best_model, best_res_new = select_best_model_result(global_results[idx]['model_results'])
                if best_res_new:
                    global_results[idx]['score'] = best_res_new['score']
                    global_results[idx]['hyp_text'] = best_res_new['hyp_text']
        else:
            global_results[idx]['verification_status'] = status
            global_results[idx]['model_used'] = 'manual'

        # Persist to CSV (best effort)
        try:
            save_df = pd.DataFrame(global_results)
            clean_name = sanitize_filename(dataset_name)
            save_path = f"{clean_name}_results.csv"
            save_df.to_csv(save_path, index=False)
            print(f"✅ Auto-saved results to {save_path}")
        except Exception as e:
            print(f"Error saving to CSV: {e}")

        return generate_dashboard_outputs(similarity_threshold)

    except Exception as e:
        print(f"Error in verify_action: {e}")
        return generate_dashboard_outputs(similarity_threshold)


def save_results_csv(dataset_name):
    global global_results
    if not global_results:
        return None

    try:
        # Падрыхтоўка даных для экспарту
        export_data = []
        for result in global_results:
            export_row = {k: v for k, v in result.items() if k not in ['audio_array', 'sampling_rate']}
            
            # Канвертаваць model_results у JSON-радок
            if 'model_results' in export_row and export_row['model_results']:
                export_row['model_results'] = json.dumps(export_row['model_results'], ensure_ascii=False)
            
            export_data.append(export_row)
        
        df_export = pd.DataFrame(export_data)
        
        # Асноўны файл
        clean_name = sanitize_filename(dataset_name)
        filename = f"{clean_name}_results.csv"
        abs_path = os.path.abspath(filename)
        df_export.to_csv(abs_path, index=False)
        print(f"💾 Exporting main CSV: {abs_path}")

        
        # Стварыць падрабязны файл з параўнаннем мадэлей
        detailed_data = []
        for result in global_results:
            model_results = result.get('model_results', {})
            if model_results:
                comparison = get_all_model_comparison(result)
                for model_name, model_result in model_results.items():
                    detailed_data.append({
                        "id": result.get('id'),
                        "path": result.get('path'),
                        "model_name": model_name,
                        "hyp_text": model_result.get('hyp_text', ''),
                        "score": model_result.get('score', 0),
                        "is_best": model_name == comparison.get('best_model', ''),
                        "ref_text": result.get('ref_text', '')
                    })
        
        if detailed_data:
            df_detailed = pd.DataFrame(detailed_data)
            detailed_filename = f"{clean_name}_model_comparison.csv"
            detailed_abs_path = os.path.abspath(detailed_filename)
            df_detailed.to_csv(detailed_abs_path, index=False)
            print(f"💾 Exporting detailed CSV: {detailed_abs_path}")
        
        return abs_path

    except Exception as e:
        print(f"Error creating CSV: {e}")
        return None


def create_verified_dataset(hf_token, dataset_name, progress=gr.Progress()):
    """
    Creates a new dataset on Hugging Face using only verified records.
    Name of the new dataset is {username}/{original_name}Checked.
    """
    global global_results
    
    if not hf_token:
        raise gr.Error("Калі ласка, увядзіце Hugging Face Token.")
    
    if not global_results:
        raise gr.Error("Няма даных для стварэння датасэта.")

    # Filter correct results
    verified_data = [r for r in global_results if r.get('verification_status') == 'correct']
    
    if not verified_data:
        raise gr.Error("Няма правераных (correct) запісаў для стварэння датасэта.")

    try:
        login(token=hf_token)
        api = HfApi(token=hf_token)
        user_info = api.whoami()
        username = user_info['name']

        # Determine new repository ID
        if "/" in dataset_name:
            original_slug = dataset_name.split("/")[-1]
        else:
            original_slug = dataset_name
            
        new_repo_id = f"{username}/{original_slug}Checked"
        
        # Generator for the new dataset
        def gen():
            # Create a map for quick lookup if we need to fall back to the original dataset
            ds_ref = None
            
            for i, row in enumerate(verified_data):
                audio_array = row.get('audio_array')
                sr = row.get('sampling_rate')
                
                # Logic to ensure we have audio
                if audio_array is None or len(audio_array) == 0:
                     if ds_ref is None:
                         progress(0, desc="Загрузка арыгінальнага датасэта для атрымання аўдыя...")
                         try:
                            # Collect all needed paths first to optimize loading
                            needed_paths = set()
                            for r in verified_data:
                                if r.get('audio_array') is None or len(r.get('audio_array')) == 0:
                                    p = r.get('path')
                                    if p:
                                        needed_paths.add(p)
                                        needed_paths.add(os.path.basename(p))
                            
                            if needed_paths:
                                full_ds_items = utils.load_hf_dataset(dataset_name, allowed_paths=needed_paths)
                                ds_map = {}
                                for item in full_ds_items:
                                    p = item['audio']['path']
                                    if p:
                                        ds_map[p] = item
                                        ds_map[os.path.basename(p)] = item
                                ds_ref = ds_map
                            else:
                                ds_ref = {} # No missing audio needed to load

                         except Exception as e:
                             print(f"Failed to load original dataset for audio fallback: {e}")
                             ds_ref = {}

                     path = row.get('path')
                     item = None
                     if ds_ref:
                        item = ds_ref.get(path) or ds_ref.get(os.path.basename(path))
                     
                     if item:
                         audio_array = item['audio']['array']
                         sr = item['audio']['sampling_rate']
                
                if audio_array is not None and len(audio_array) > 0:
                     # Manually encode to WAV bytes to avoid dependency on torch/librosa within datasets library
                     buffer = io.BytesIO()
                     # Ensure sampling_rate is integer
                     safe_sr = int(float(sr)) if sr else 16000
                     sf.write(buffer, audio_array, safe_sr, format='WAV')
                     audio_bytes = buffer.getvalue()
                     
                     yield {
                         "audio": {"bytes": audio_bytes, "path": None},
                         "text": row.get('ref_text', ''),
                         "original_path": row.get('path', '')
                     }
        
        # Define features using primitive types to avoid 'Audio' feature triggering torch checks
        features = Features({
            "audio": {"bytes": Value("binary"), "path": Value("string")}, 
            "text": Value("string"),
            "original_path": Value("string")
        })
        
        new_ds = Dataset.from_generator(gen, features=features)

        # Manually patch the feature metadata to 'Audio' so HF Hub recognizes it as audio
        # This bypasses the strict dependency checks in datasets.Audio.encode_example
        if "audio" in new_ds.features:
            new_ds.info.features["audio"] = Audio(sampling_rate=None)
        
        if len(new_ds) == 0:
             raise gr.Error("Не ўдалося сабраць аўдыяданыя для правераных запісаў.")

        progress(0.9, desc=f"Загрузка датасэта '{new_repo_id}' на Hugging Face...")
        
        new_ds.push_to_hub(new_repo_id, token=hf_token)
        
        return f"✅ Датасэт паспяхова створаны: https://huggingface.co/datasets/{new_repo_id}"

    except Exception as e:
        raise gr.Error(f"Памылка стварэння датасэта: {e}")


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

  // localStorage HF Token persistence
  function setupHfTokenSaver() {
    const el = getValueEl("hf_token_input");
    if (!el) return false;

    const saved = localStorage.getItem("hf_token");
    if (saved && !el.value) {
      el.value = saved;
      el.dispatchEvent(new Event("input", { bubbles: true }));
      el.dispatchEvent(new Event("change", { bubbles: true }));
    }

    if (!el.__hfSaverAttached) {
      el.addEventListener("blur", function () {
        if (this.value && this.value.length > 5) {
          localStorage.setItem("hf_token", this.value);
        }
      });
      el.__hfSaverAttached = true;
    }
    return true;
  }

  // retry a few times because Gradio renders async
  let tries = 0;
  const timer = setInterval(() => {
    tries += 1;
    const ok = setupApiKeySaver();
    const ok2 = setupHfTokenSaver();
    if ((ok && ok2) || tries >= 40) clearInterval(timer);
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

            hf_token = gr.Textbox(
                label="Hugging Face Token",
                type="password",
                placeholder="Увядзіце ваш HF Token (для стварэння датасэту)...",
                elem_id="hf_token_input"
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



            batch_mode = gr.Checkbox(
                label="Пакетная апрацоўка (Batch Mode)",
                value=False,
                info="Калі ўключана, выкарыстоўвае Gemini Batch API (танней, але павольней)."
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
            
            with gr.Row():
                create_ds_btn = gr.Button("🤗 Стварыць праверыны датасэт", variant="primary")
                create_ds_output = gr.Markdown()

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
        inputs=[api_key, dataset_name, model_name, limit_files, temperature, thinking_budget, similarity_threshold, batch_mode, recheck_problematic],
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
        inputs=[dataset_name],
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

    create_ds_btn.click(
        fn=create_verified_dataset,
        inputs=[hf_token, dataset_name],
        outputs=[create_ds_output]
    )

    # Verification event
    verification_trigger.click(
        fn=verify_action,
        inputs=[verification_data, similarity_threshold, dataset_name],
        outputs=[stats_output, flagged_output, results_table]
    )

if __name__ == "__main__":
    demo.launch(debug=True)
