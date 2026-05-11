import threading
from transformers import pipeline

from src.app.core.config import settings


_pipe = None
_lock = threading.Lock()


def get_whisper_pipeline():
    """Lazy-load Whisper model."""
    global _pipe
    if _pipe is None:
        with _lock:
            if _pipe is None:
                _pipe = pipeline(model=settings.whisper_model)
    return _pipe
