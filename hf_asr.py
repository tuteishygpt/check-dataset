"""Hugging Face ASR Integration for Belarusian Speech Recognition."""
import os
import tempfile
import soundfile as sf
from gradio_client import Client, handle_file


# Available HF ASR models
HF_ASR_MODELS = {
    "SeamlessM4T-v2 (HF)": {
        "space_id": "archivartaunik/ASR_BEL_SeamlessM4Tv2_Batch",
        "api_name": "/transcribe_many_ui",
        "description": "SeamlessM4T v2 для беларускай мовы"
    }
}


class HuggingFaceASR:
    """Client for Hugging Face ASR Spaces."""
    
    def __init__(self, space_id: str):
        """Initialize the HF ASR client.
        
        Args:
            space_id: The Hugging Face Space ID (e.g., "archivartaunik/ASR_BEL_SeamlessM4Tv2_Batch")
        """
        self.space_id = space_id
        self.client = None
    
    def _ensure_client(self):
        """Lazily initialize the Gradio client."""
        if self.client is None:
            self.client = Client(self.space_id)
        return self.client
    
    def transcribe_audio(self, audio_array, sampling_rate: int) -> str:
        """Transcribe a single audio file.
        
        Args:
            audio_array: NumPy array of audio data
            sampling_rate: Sample rate of the audio
            
        Returns:
            Transcribed text
        """
        client = self._ensure_client()
        
        # Save audio to temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_path = tmp_file.name
            sf.write(tmp_path, audio_array, int(sampling_rate), format='WAV')
        
        try:
            # Call the API with single file in list
            result = client.predict(
                files=[handle_file(tmp_path)],
                api_name="/transcribe_many_ui"
            )
            
            # Result is a tuple: (dataframe_dict, csv_filepath, status_string)
            # dataframe_dict has 'headers' and 'data' keys
            if result and len(result) >= 1:
                df_dict = result[0]
                if df_dict and 'data' in df_dict and len(df_dict['data']) > 0:
                    # First row, get the transcription column (usually last column)
                    row = df_dict['data'][0]
                    # The data format is typically [filename, transcription]
                    if len(row) >= 2:
                        return str(row[-1])  # Last column is transcription
                    elif len(row) == 1:
                        return str(row[0])
            
            return ""
            
        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    def transcribe_batch(self, audio_files: list) -> dict:
        """Transcribe multiple audio files in a batch.
        
        Args:
            audio_files: List of tuples (key, audio_array, sampling_rate)
            
        Returns:
            Dictionary mapping keys to transcribed text
        """
        client = self._ensure_client()
        
        # Create temp files for all audio
        temp_files = []
        key_to_filename = {}
        
        try:
            for key, audio_array, sampling_rate in audio_files:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                    tmp_path = tmp_file.name
                    sf.write(tmp_path, audio_array, int(sampling_rate), format='WAV')
                    temp_files.append(tmp_path)
                    key_to_filename[os.path.basename(tmp_path)] = key
            
            # Call the API with all files
            result = client.predict(
                files=[handle_file(f) for f in temp_files],
                api_name="/transcribe_many_ui"
            )
            
            # Parse results
            transcriptions = {}
            if result and len(result) >= 1:
                df_dict = result[0]
                if df_dict and 'data' in df_dict:
                    for row in df_dict['data']:
                        if len(row) >= 2:
                            filename = str(row[0])
                            text = str(row[-1])
                            # Match filename back to key
                            for temp_file in temp_files:
                                if os.path.basename(temp_file) in filename or filename in os.path.basename(temp_file):
                                    key = key_to_filename.get(os.path.basename(temp_file))
                                    if key:
                                        transcriptions[key] = text
                                        break
            
            return transcriptions
            
        finally:
            # Clean up temp files
            for tmp_path in temp_files:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)


def get_hf_asr_client(model_name: str) -> HuggingFaceASR:
    """Get an HF ASR client for the given model name.
    
    Args:
        model_name: Name of the model from HF_ASR_MODELS
        
    Returns:
        HuggingFaceASR client instance
    """
    if model_name not in HF_ASR_MODELS:
        raise ValueError(f"Unknown HF ASR model: {model_name}")
    
    model_config = HF_ASR_MODELS[model_name]
    return HuggingFaceASR(model_config["space_id"])


def is_hf_asr_model(model_name: str) -> bool:
    """Check if the model name is a Hugging Face ASR model."""
    return model_name in HF_ASR_MODELS


def get_all_asr_model_choices() -> list:
    """Get list of all available ASR models (Gemini + HF)."""
    gemini_models = [
        "gemini-3-pro-preview",
        "gemini-3-flash-preview",
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
    ]
    hf_models = list(HF_ASR_MODELS.keys())
    return gemini_models + hf_models
