import edge_tts
import asyncio

async def generate_polish_audio(text, output_file):
    tts = edge_tts.Communicate(text=text, voice="pl-PL-MarekNeural")
    await tts.save(output_file)
