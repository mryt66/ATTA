import edge_tts

from src.app.core.config import settings


async def generate_speech(text: str, output_file: str) -> None:
    """Generate speech audio from text using Edge TTS."""
    tts = edge_tts.Communicate(text=text, voice=settings.tts_voice)
    await tts.save(output_file)
