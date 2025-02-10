import tempfile
import os
import requests
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Body
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import language_tool_python
import json
from utils.correction import transcribe_audio_file, pipe1, tool1

app = FastAPI()

@app.on_event("startup")
def load_model():
    """Load model and language tool on startup"""
    global pipe, tool
    pipe = pipe1
    tool = tool1

@app.post("/transcribe/")
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

@app.post("/chat_llm/")
async def chat(text: str = Body(..., embed=True)):
    url = "http://localhost:11434/api/chat"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": "SpeakLeash/bielik-11b-v2.3-instruct:Q6_K",
        "messages": [
            {"role": "user", "content": text},
        ],
        "stream": False,
        "max_length": 10
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

# @app.post("/tts/")
# async def tts(text: str = Form(...)):

# @app.post("/transcribe-chat/")
# async def transcribe_and_chat(file: UploadFile = File(...)):
#     if not file.filename.lower().endswith(('.mp3', '.wav')):
#         raise HTTPException(400, "Only MP3/WAV files are allowed")
#     file_ext = os.path.splitext(file.filename)[1]
#     with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
#         content = await file.read()
#         tmp_file.write(content)
#         tmp_path = tmp_file.name
#     try:
#         transcription = transcribe_audio_file(tmp_path)
#     except Exception as e:
#         os.remove(tmp_path)
#         raise HTTPException(500, f"Transcription error: {str(e)}")
#     finally:
#         os.remove(tmp_path)
#     url = "http://localhost:11434/api/chat"
#     headers = {"Content-Type": "application/json"}
#     data = {
#         "model": "SpeakLeash/bielik-11b-v2.3-instruct:Q4_K_M",
#         "messages": [
#             {"role": "user", "content": transcription},
#         ],
#         "stream": False
#     }
#     try:
#         response = requests.post(url, headers=headers, data=json.dumps(data))
#         chat_result = response.json()
#         chat_response = chat_result["message"]["content"]
#     except Exception as e:
#         raise HTTPException(500, f"Chat API error: {str(e)}")
#     return {"transcription": transcription, "chat_response": chat_response}