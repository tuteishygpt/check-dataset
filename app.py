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
from core.env import configure_environment

# Load environment variables before importing modules that inspect temp/cache paths.
configure_environment()

from core.state import clear_analysis_logs, install_global_log_capture
from ui.gradio_app import create_interface


def main():
    """Launch the application."""
    install_global_log_capture()
    clear_analysis_logs()
    demo = create_interface()
    demo.launch(debug=True)


if __name__ == "__main__":
    main()
