"""Dataset caching utilities."""
import hashlib
from core.state import get_dataset_cache


def get_dataset_cache_key(dataset_name: str, limit: int) -> str:
    """Generate a cache key for the dataset."""
    return hashlib.md5(f"{dataset_name}:{limit}".encode()).hexdigest()


def get_cached_dataset(dataset_name: str, limit: int):
    """Get cached dataset if available."""
    cache_key = get_dataset_cache_key(dataset_name, limit)
    return get_dataset_cache().get(cache_key)


def cache_dataset(dataset_name: str, limit: int, data):
    """Cache the downloaded dataset."""
    cache_key = get_dataset_cache_key(dataset_name, limit)
    get_dataset_cache()[cache_key] = data
