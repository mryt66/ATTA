import asyncio
import tempfile
import sounddevice as sd
from pydub import AudioSegment
from scipy.io.wavfile import write
from src.services.stt import transcribe_audio_file
from src.services.llm import chat_with_llm
from src.services.tts import generate_speech


SAMPLE_RATE = 44100
DURATION = 5
WAV_FILENAME = "temp.wav"
MP3_FILENAME = "temp.mp3"


def record_audio() -> str:
    """Record audio and return path to MP3 file."""
    print("Recording...")
    audio_data = sd.rec(
        int(SAMPLE_RATE * DURATION), samplerate=SAMPLE_RATE, channels=1, dtype="int16"
    )
    sd.wait()
    print("End of recording")
    write(WAV_FILENAME, SAMPLE_RATE, audio_data)

    audio = AudioSegment.from_wav(WAV_FILENAME)
    audio.export(MP3_FILENAME, format="mp3")
    return MP3_FILENAME


async def main():
    audio_file = record_audio()

    transcription = transcribe_audio_file(audio_file)
    print(f"Transcription: {transcription}")

    llm_response = chat_with_llm(transcription)
    print(f"LLM Response: {llm_response}")

    await generate_speech(llm_response, "output.mp3")
    print("Audio saved to output.mp3")


if __name__ == "__main__":
    asyncio.run(main())
