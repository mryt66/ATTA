import tempfile
import os
import requests
from fastapi.responses import JSONResponse
from pydub import AudioSegment
import base64
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Body
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from utils.correction import transcribe_audio_file, pipe1, tool1

app = FastAPI(
    title="Voice Assistant ATTA API",
    description="API for audio to text to audio processing",
    version="0.1.0"
)

@app.on_event("startup")
def load_model():
    """Load model and language tool on startup"""
    global pipe, tool
    pipe = pipe1
    tool = tool1

@app.post("/transcribe/", summary="Transcribe an audio file")
async def handle_transcription(file: UploadFile = File(...)):
    """Endpoint for handling audio file uploads"""
    if not file.filename.lower().endswith(('.mp3', '.wav')):
        raise HTTPException(400, "Only MP3/WAV files are allowed")
    
    file_ext = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_path = tmp_file.name
    
    try:
        transcription = transcribe_audio_file(tmp_path)
    except Exception as e:
        os.remove(tmp_path)
        raise HTTPException(500, f"Processing error: {str(e)}")
    finally:
        os.remove(tmp_path)
    
    return {"transcription": transcription}

@app.post("/chat_llm/", summary="Chat with a language model")
async def chat(text: str = Body(..., embed=True)):
    url = "http://localhost:11434/api/chat"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": "hf.co/marcsixtysix/gemma-3-1b-it-pl-polqa",
        "messages": [
            {"role": "user", "content": text},
        ],
        "stream": False,
        "max_length": 128
    }
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        
        if "message" in result and "content" in result["message"]:
            return {"response": result["message"]["content"]}
        else:
            raise HTTPException(status_code=500, detail="Invalid response format from LLM")

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Error connecting to LLM: {str(e)}")

    return result["message"]["content"]

@app.post("/tts/")
async def tts_endpoint(
    text: str = Form(...),
    lang: str = Form("en"),
    format: str = Form("wav")
):
    sample_rate = 16000
    channels = 1
    sample_width = 2 
    audio_segment = AudioSegment.silent(duration=1000, frame_rate=sample_rate).set_channels(channels).set_sample_width(sample_width)

    audio_bytes = audio_segment.raw_data
    audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
    
    return JSONResponse(content={
        "samples": audio_base64,
        "sample_rate": sample_rate,
        "channels": channels,
        "sample_width": sample_width
    })
