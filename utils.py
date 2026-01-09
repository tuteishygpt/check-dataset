import re
import io
import time
import random
from rapidfuzz import fuzz
from google import genai
from datasets import load_dataset, Audio
import soundfile as sf
import numpy as np
import librosa

def load_hf_dataset(dataset_name, split="train", limit=None, allowed_paths=None):
    """
    Loads the dataset from Hugging Face with manual audio loading.
    Avoids torchcodec dependency by loading audio via librosa.
    """
    try:
        # Load dataset with audio decoding disabled
        ds = load_dataset(dataset_name, split=split)
        
        # Cast audio column to disable automatic decoding (returns raw bytes)
        if 'audio' in ds.features:
            ds = ds.cast_column('audio', Audio(decode=False))
        
        if limit:
            ds = ds.select(range(min(limit, len(ds))))
        
        # Process each item to load audio manually
        processed_items = []
        for item in ds:
             # Filter by allowed_paths if provided
            if allowed_paths is not None:
                audio_info_check = item.get('audio', {})
                if isinstance(audio_info_check, dict):
                    path_check = audio_info_check.get('path')
                    if not path_check:
                         continue
                    if path_check not in allowed_paths and os.path.basename(path_check) not in allowed_paths:
                        continue
            
            processed_item = dict(item)
            
            # Handle audio loading manually from raw bytes
            if 'audio' in item:
                audio_info = item['audio']
                if isinstance(audio_info, dict):
                    audio_bytes_data = audio_info.get('bytes')
                    audio_path = audio_info.get('path', 'unknown')
                    
                    if audio_bytes_data:
                        # Audio is in bytes format - load with librosa
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

def transcribe_audio(client, model_name, audio_array, sampling_rate, config=None, max_retries=5):
    """
    Transcribes audio using Gemini API with optional config.
    Implements retry logic with exponential backoff for 429 rate limit errors.
    
    Args:
        client: Gemini API client
        model_name: Name of the model to use
        audio_array: Audio data as numpy array
        sampling_rate: Audio sampling rate
        config: Optional generation config
        max_retries: Maximum number of retry attempts for rate limit errors
    
    Returns:
        Transcribed text or error message
    """
    # Convert numpy array to bytes (WAV format)
    # Gemini expects bytes or a file URI. Since we have the data in memory from 'datasets',
    # we convert it to a WAV buffer.
    
    # Create an in-memory byte buffer
    audio_buffer = io.BytesIO()
    try:
        sr = int(float(sampling_rate)) if sampling_rate is not None else 16000
    except (ValueError, TypeError):
        sr = 16000
    sf.write(audio_buffer, audio_array, sr, format='WAV')
    audio_bytes = audio_buffer.getvalue()
    
    last_error = None
    
    for attempt in range(max_retries):
        try:
            # Generate content
            response = client.models.generate_content(
                model=model_name,
                contents=[
                    genai.types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav"),
                    "Transcribe the following audio exactly as spoken in belarusian. Output ONLY the transcription."
                ],
                config=config
            )
            
            return response.text.strip()
            
        except Exception as e:
            error_str = str(e)
            last_error = e
            
            # Check if this is a 429 RESOURCE_EXHAUSTED error
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                # Try to parse retry delay from error message
                # Example: "Please retry in 54.450133291s"
                wait_time = 60  # Default wait time in seconds
                
                import re
                retry_match = re.search(r'retry in (\d+(?:\.\d+)?)s', error_str)
                if retry_match:
                    wait_time = float(retry_match.group(1)) + random.uniform(1, 5)  # Add jitter
                else:
                    # Exponential backoff: 60s, 120s, 240s, ...
                    wait_time = 60 * (2 ** attempt) + random.uniform(1, 10)
                
                if attempt < max_retries - 1:
                    print(f"⏳ Квота вычарпана (спроба {attempt + 1}/{max_retries}). "
                          f"Чаканне {wait_time:.1f} сек...")
                    time.sleep(wait_time)
                    continue
            else:
                # For non-rate-limit errors, don't retry
                return f"Error: {e}"
    
    # All retries exhausted
    return f"Error: Вычарпана максімальная колькасць спроб. Апошняя памылка: {last_error}"

