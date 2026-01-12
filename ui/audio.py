"""Audio utilities for UI."""
import io
import base64
import pandas as pd
import soundfile as sf
import numpy as np


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


def get_audio_for_row(global_results, row_index: int):
    """Get audio data for a specific row."""
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
