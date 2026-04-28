"""Global state management for the application."""

from contextlib import contextmanager
import builtins
import logging
import sys
import threading

# Global variable to store results for audio playback
global_results = []
_results_lock = threading.Lock()

# Cache for downloaded datasets
dataset_cache = {}
_cache_lock = threading.Lock()

# Analysis log buffer
analysis_logs = []
analysis_logs_lock = threading.Lock()
analysis_running = False
_analysis_running_lock = threading.Lock()
log_capture_installed = False
original_stdout = sys.stdout
original_stderr = sys.stderr
analysis_log_handler = None

# Stop flag for long-running tasks
stop_requested = False
_stop_lock = threading.Lock()


def get_stop_requested():
    """Check if a stop has been requested."""
    with _stop_lock:
        return stop_requested


def set_stop_requested(value: bool):
    """Set the stop requested flag."""
    global stop_requested
    with _stop_lock:
        stop_requested = value


def get_global_results():
    """Get the global results list."""
    with _results_lock:
        return global_results


def set_global_results(results):
    """Set the global results list."""
    global global_results
    with _results_lock:
        global_results = results


def clear_global_results():
    """Clear all global results."""
    global global_results
    with _results_lock:
        global_results = []


def append_analysis_log(message):
    """Append a line to the shared analysis log buffer."""
    if message is None:
        return

    text = str(message).rstrip()
    if not text:
        return

    global analysis_logs
    with analysis_logs_lock:
        analysis_logs.extend(text.splitlines())
        analysis_logs = analysis_logs[-2000:]


def get_analysis_logs():
    """Return a copy of analysis log lines."""
    with analysis_logs_lock:
        return list(analysis_logs)


def get_analysis_logs_text():
    """Return analysis logs as newline-separated text."""
    return "\n".join(get_analysis_logs())


def clear_analysis_logs():
    """Clear the shared analysis log buffer."""
    global analysis_logs
    with analysis_logs_lock:
        analysis_logs = []


def set_analysis_running(value: bool):
    """Set whether an analysis task is currently running."""
    global analysis_running
    with _analysis_running_lock:
        analysis_running = bool(value)


def get_analysis_running():
    """Return whether an analysis task is currently running."""
    with _analysis_running_lock:
        return analysis_running


class _LogStreamProxy:
    """Mirror writes to the original stream and the in-app log buffer."""

    def __init__(self, target):
        self.target = target
        self._pending = ""
        self._lock = threading.Lock()

    @property
    def encoding(self):
        return getattr(self.target, "encoding", None)

    def write(self, text):
        if text is None:
            return 0

        message = str(text)
        if not message:
            return 0

        self.target.write(message)
        self._capture(message)
        return len(message)

    def flush(self):
        self.target.flush()
        with self._lock:
            if self._pending:
                append_analysis_log(self._pending)
                self._pending = ""

    def isatty(self):
        return bool(getattr(self.target, "isatty", lambda: False)())

    def writable(self):
        return True

    def _capture(self, text: str):
        with self._lock:
            combined = self._pending + text
            parts = combined.splitlines(keepends=True)
            self._pending = ""

            for part in parts:
                if part.endswith("\n") or part.endswith("\r"):
                    append_analysis_log(part.rstrip("\r\n"))
                else:
                    self._pending += part


class _AnalysisLogHandler(logging.Handler):
    """Send Python logging records to the shared in-app log buffer."""

    def emit(self, record):
        try:
            append_analysis_log(self.format(record))
        except Exception:
            pass


def install_global_log_capture():
    """Install process-wide stdout/stderr/logging capture for the UI log panel."""
    global log_capture_installed, analysis_log_handler, original_stdout, original_stderr

    if log_capture_installed:
        return

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = _LogStreamProxy(original_stdout)
    sys.stderr = _LogStreamProxy(original_stderr)

    handler = _AnalysisLogHandler()
    handler.setLevel(logging.NOTSET)
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )
    logging.getLogger().addHandler(handler)
    logging.getLogger().setLevel(logging.NOTSET)
    logging.captureWarnings(True)

    analysis_log_handler = handler
    log_capture_installed = True


@contextmanager
def capture_analysis_prints():
    """Mirror print output into the shared analysis log buffer."""
    original_print = builtins.print

    def tee_print(*args, **kwargs):
        sep = kwargs.get("sep", " ")
        end = kwargs.get("end", "\n")
        message = sep.join(str(arg) for arg in args)
        append_analysis_log(f"{message}{end}")
        return original_print(*args, **kwargs)

    builtins.print = tee_print
    try:
        yield
    finally:
        builtins.print = original_print


def get_dataset_cache():
    """Get the dataset cache dictionary."""
    with _cache_lock:
        return dataset_cache


def clear_dataset_cache():
    """Clear the dataset cache."""
    global dataset_cache
    with _cache_lock:
        count = len(dataset_cache)
        dataset_cache.clear()
        return count
