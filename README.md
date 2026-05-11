# ATTA — Voice Chatbot API

ATTA is a voice chatbot API that transcribes audio (speech-to-text), processes it with an LLM, and returns a spoken response (text-to-speech). Designed for Polish language interaction.

## Requirements

- Python 3.12+
- [Ollama](https://ollama.com/) running locally with model `hf.co/marcsixtysix/gemma-3-1b-it-pl-polqa`
- ffmpeg (required by pydub for audio processing)

## Setup

```bash
# Pull the LLM model
ollama pull hf.co/marcsixtysix/gemma-3-1b-it-pl-polqa

# Install dependencies
uv pip install -e .
```

## Demo

<video src="./assets/demo_atta.mp4" controls width="600"></video>

## Run

```bash
uvicorn src.app.api:app --reload
```

The API is available at `http://localhost:8000`.

## Endpoints

| Method | Path            | Description                          |
|--------|-----------------|--------------------------------------|
| GET    | `/`             | Voice chat web UI                    |
| GET    | `/health`       | Health check                         |
| POST   | `/transcribe/`  | Transcribe an uploaded WAV file      |
| POST   | `/chat_llm/`    | Chat with the LLM (text in, text out)|
| POST   | `/tts/`         | Text-to-speech (text in, audio out)  |
| POST   | `/voice_chat/`  | Full pipeline: STT → LLM → TTS      |
| POST   | `/reset_chat/`  | Reset conversation history           |
