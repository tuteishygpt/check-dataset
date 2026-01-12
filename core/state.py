"""Global state management for the application."""

# Global variable to store results for audio playback
global_results = []

# Cache for downloaded datasets
dataset_cache = {}


def get_global_results():
    """Get the global results list."""
    global global_results
    return global_results


def set_global_results(results):
    """Set the global results list."""
    global global_results
    global_results = results


def clear_global_results():
    """Clear all global results."""
    global global_results
    global_results = []


def get_dataset_cache():
    """Get the dataset cache dictionary."""
    global dataset_cache
    return dataset_cache


def clear_dataset_cache():
    """Clear the dataset cache."""
    global dataset_cache
    count = len(dataset_cache)
    dataset_cache.clear()
    return count
