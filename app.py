"""
TTS Dataset Validator - Main Application Entry Point

This application analyzes audio datasets to detect mismatches between 
metadata and audio content using Gemini on Vertex AI.

Modular structure:
- core/: State management, caching, model comparison
- analysis/: Standard analysis, smart analysis, import/export
- ui/: Gradio interface, dashboard, styles, audio utilities
- gemini_api.py: Vertex AI integration
- utils.py: Text processing and dataset utilities
"""
from dotenv import load_dotenv
from ui.gradio_app import create_interface

# Load environment variables
load_dotenv()


def main():
    """Launch the application."""
    demo = create_interface()
    demo.launch(debug=True)


if __name__ == "__main__":
    main()
