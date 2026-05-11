from src.app.core.model_loader import get_whisper_pipeline


def transcribe_audio_file(file_path: str) -> str:
    """Transcribe audio file and return text."""
    pipe = get_whisper_pipeline()
    from src.app.core.config import settings
    result = pipe(
        file_path,
        generate_kwargs={"language": settings.language, "no_repeat_ngram_size": 3},
    )
    return result["text"]
