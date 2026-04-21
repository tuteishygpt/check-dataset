"""Helpers for handling Hugging Face authentication input."""


def normalize_hf_token(hf_token):
    """Normalize blank UI input to a missing token."""
    if hf_token is None:
        return None

    normalized = str(hf_token).strip()
    return normalized or None
