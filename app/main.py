import sounddevice as sd
from scipy.io.wavfile import write
from pydub import AudioSegment
from utils.requests import ATTAPI
import time
import asyncio

from utils.tts import generate_polish_audio


SAMPLE_RATE = 44100
DURATION = 5  
WAV_FILENAME = "temp.wav"
MP3_FILENAME = "temp.mp3"

print("Recording...")
audio_data = sd.rec(int(SAMPLE_RATE * DURATION), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
sd.wait()
print("End of recording")
write(WAV_FILENAME, SAMPLE_RATE, audio_data)

audio = AudioSegment.from_wav(WAV_FILENAME)
audio.export(MP3_FILENAME, format="mp3")

url = "https://meet-tahr-radically.ngrok-free.app/"
atta_object = ATTAPI(url)
time1 = time.time()
response = atta_object.transcribe(MP3_FILENAME)
print(f"Time for transcription: {time.time()-time1}")
print(response["transcription"])

time2 = time.time()
response_llm = atta_object.llm(response["transcription"])
print(f"Time for LLM response: {time.time()-time2}")
print(response_llm)


await generate_polish_audio(response_llm, "output.mp3")