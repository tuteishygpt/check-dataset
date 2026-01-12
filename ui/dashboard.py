"""Dashboard HTML generation."""
import html
import pandas as pd

from core.state import get_global_results
from core.comparison import get_all_model_comparison, find_best_model_pair
from ui.audio import array_to_b64_audio


def _e(x) -> str:
    """HTML-escape helper."""
    return html.escape("" if x is None else str(x), quote=True)


def generate_dashboard_outputs(similarity_threshold: int):
    """
    Generates the HTML/DF outputs for the dashboard based on global_results.
    Refactored to be used by all analysis functions.
    """
    global_results = get_global_results()

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
                
            flagged_html += _generate_flagged_item_html(row, similarity_threshold)

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


def _generate_flagged_item_html(row, similarity_threshold: int) -> str:
    """Generate HTML for a single flagged item."""
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

    return f"""
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
