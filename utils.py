import re
import os
import io
import time
import random
from rapidfuzz import fuzz
from google import genai
from datasets import load_dataset, Audio
import soundfile as sf
import numpy as np
import librosa

def load_hf_dataset(dataset_name, split="train", limit=None, allowed_paths=None, hf_token=None):
    """
    Loads the dataset from Hugging Face with manual audio loading.
    Avoids torchcodec dependency by loading audio via librosa.
    """
    try:
        # Load dataset with audio decoding disabled
        ds = load_dataset(dataset_name, split=split, token=hf_token)
        
        # Cast audio column to disable automatic decoding (returns raw bytes)
        if 'audio' in ds.features:
            ds = ds.cast_column('audio', Audio(decode=False))
        
        if limit:
            ds = ds.select(range(min(limit, len(ds))))
        
        # 1. Спачатку збіраем усе метаданыя без загрузкі аўдыя-байтаў
        unique_items_map = {}
        total_ds_count = len(ds)
        print(f"🔍 Загружана метаданых: {total_ds_count} запісаў")

        for item in ds:
            # Вызначаем імя файла для дэдублікацыі
            audio_info = item.get('audio', {})
            audio_path = audio_info.get('path', 'unknown') if isinstance(audio_info, dict) else 'unknown'
            file_name = os.path.basename(audio_path)

            if audio_path == 'unknown':
                # Калі шлях невядомы, проста дадаем (не варта дэдублікаваць па 'unknown')
                 unique_items_map[f"unknown_{len(unique_items_map)}"] = item
            else:
                # Падтрымліваем толькі апошнюю версію па імені файла
                unique_items_map[file_name] = item

        # 2. Прымяняем ліміт пасля дэдублікацыі
        final_unique_list = list(unique_items_map.values())
        unique_count = len(final_unique_list)
        print(f"✨ Пасля выдалення дубляў засталося: {unique_count} унікальных файлаў")

        if limit and limit > 0:
            final_unique_list = final_unique_list[:limit]
            print(f"✂️ Прыменены ліміт: пакінута {len(final_unique_list)} файлаў")

        # 3. Цяпер загружаем аўдыя толькі для выбраных унікальных файлаў
        processed_items = []
        for i, item in enumerate(final_unique_list):
            # Фільтрацыя па дазволеных шляхах (калі ёсць)
            if allowed_paths is not None:
                audio_info = item.get('audio', {})
                path_check = audio_info.get('path') if isinstance(audio_info, dict) else None
                if not path_check: continue
                if path_check not in allowed_paths and os.path.basename(path_check) not in allowed_paths:
                    continue

            processed_item = dict(item)
            
            # Загрузка і дэкадаванне аўдыя
            if 'audio' in item:
                audio_info = item['audio']
                if isinstance(audio_info, dict):
                    audio_bytes_data = audio_info.get('bytes')
                    audio_path = audio_info.get('path', 'unknown')
                    
                    if audio_bytes_data:
                        # Загрузка праз librosa
                        audio_buffer = io.BytesIO(audio_bytes_data)
                        audio_array, sr = librosa.load(audio_buffer, sr=None)
                    else:
                        audio_array, sr = np.array([]), 16000
                    
                    processed_item['audio'] = {
                        'array': audio_array,
                        'sampling_rate': sr,
                        'path': audio_path
                    }
            
            processed_items.append(processed_item)

        print(f"✅ Гатова да апрацоўкі: {len(processed_items)} файлаў")
        return processed_items
    except Exception as e:
        raise RuntimeError(f"Error loading dataset: {e}")

def normalize_text(text):
    """
    Removes punctuation and converts to lowercase.
    """
    if not isinstance(text, str):
        return ""
    # Remove punctuation using regex, keep spaces and alphanumeric (including Cyrillic)
    # \w matches any word character (equivalent to [a-zA-Z0-9_])
    # We want to remove standard punctuation characters.
    # A simple approach for Belarusian is to keep words and spaces.
    
    # Remove all characters that are NOT word characters or whitespace
    text = re.sub(r'[^\w\s]', '', text) 
    # Also remove underscores as they are technically 'word characters' but usually unwanted in this context
    text = text.replace('_', ' ')
    
    # Compress multiple spaces to one
    text = re.sub(r'\s+', ' ', text)
    
    return text.lower().strip()

def calculate_similarity(reference, hypothesis):
    """
    Calculates the Levenshtein Ratio between reference and hypothesis.
    Returns a score between 0 and 100.
    """
    norm_ref = normalize_text(reference)
    norm_hyp = normalize_text(hypothesis)
    
    # fuzz.ratio calculates the Levenshtein Distance
    score = fuzz.ratio(norm_ref, norm_hyp)
    return score, norm_ref, norm_hyp



