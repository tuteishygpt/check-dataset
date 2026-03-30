"""Global state management for the application."""

# Global variable to store results for audio playback
global_results = []

# Cache for downloaded datasets
dataset_cache = {}


# Stop flag for long-running tasks
stop_requested = False


def get_stop_requested():
    """Check if a stop has been requested."""
    global stop_requested
    return stop_requested


def set_stop_requested(value: bool):
    """Set the stop requested flag."""
    global stop_requested
    stop_requested = value


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
