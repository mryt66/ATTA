import base64
import os
import tempfile
from datetime import datetime
from pathlib import Path
from uuid import uuid4
from contextlib import asynccontextmanager
from threading import Lock
import httpx
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Body, Header
from fastapi.responses import HTMLResponse, JSONResponse
from pydub import AudioSegment
from src.services.stt import transcribe_audio_file
from src.services.llm import chat_with_llm_messages

_session_lock = Lock()
_session_messages: dict[str, list[dict[str, str]]] = {}


def _get_session_id(session_id: str | None) -> str:
    return session_id or "default"


def _get_messages(session_id: str) -> list[dict[str, str]]:
    with _session_lock:
        return list(_session_messages.get(session_id, []))


def _append_message(session_id: str, role: str, content: str) -> None:
    with _session_lock:
        _session_messages.setdefault(session_id, []).append(
            {"role": role, "content": content}
        )


def _ensure_polish_system_prompt(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    if messages and messages[0].get("role") == "system":
        return messages
    system_message = {
        "role": "system",
        "content": "Jesteś ekseprtem odpowiadającym na pytania. Odpowiadaj wyłącznie po polsku. Nie używaj żadnego innego języka.",
    }
    return [system_message, *messages]


def _reset_messages(session_id: str) -> None:
    with _session_lock:
        _session_messages[session_id] = []


def _get_recordings_dir() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    recordings_dir = project_root / "inputs"
    recordings_dir.mkdir(parents=True, exist_ok=True)
    return recordings_dir


@asynccontextmanager
async def lifespan(_app: FastAPI):
    try:
        get_whisper_pipeline = __import__("src.app.core.model_loader", fromlist=["get_whisper_pipeline"]).get_whisper_pipeline
        get_whisper_pipeline()
    except Exception:
        # If preload fails, continue; request will try again.
        pass
    yield


app = FastAPI(
    title="Voice Assistant ATTA API",
    description="API for audio to text to audio processing",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/", response_class=HTMLResponse)
def index():
    return """
<!doctype html>
<html lang=\"en\">
  <head>
    <meta charset=\"utf-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
    <title>ATTA Voice Bot</title>
    <style>
      body { font-family: "Georgia", "Times New Roman", serif; background: #f7f1e9; color: #2a1f15; margin: 0; }
      .wrap { max-width: 720px; margin: 40px auto; padding: 24px; background: #fff7ed; border: 1px solid #e2d4c4; box-shadow: 0 8px 24px rgba(42, 31, 21, 0.08); }
      h1 { font-size: 28px; margin: 0 0 8px; }
      p { margin: 0 0 20px; color: #5c4633; }
      .controls { display: flex; gap: 12px; margin-bottom: 16px; flex-wrap: wrap; }
      button { padding: 10px 18px; font-size: 16px; border-radius: 8px; border: 1px solid #c9b39a; background: #f0dfc8; cursor: pointer; }
      button:disabled { opacity: 0.5; cursor: not-allowed; }
      .status { font-size: 14px; color: #7a5f49; margin-bottom: 12px; }
      .panel { background: #fff; border: 1px solid #e2d4c4; padding: 12px; border-radius: 8px; }
      .label { font-weight: bold; margin-bottom: 6px; display: block; }
      .log { background: #fef9f2; border: 1px solid #eadac8; padding: 10px; border-radius: 8px; min-height: 80px; max-height: 200px; overflow-y: auto; font-size: 14px; color: #4a392b; }
      .log-entry { margin-bottom: 8px; }
      .log-role { font-weight: bold; }
      audio { width: 100%; margin-top: 8px; }
    </style>
  </head>
  <body>
    <div class=\"wrap\">
      <h1>ATTA Voice Bot</h1>
      <p>Record a message and get a spoken response.</p>
      <div class=\"controls\">
        <button id=\"recordBtn\">Record</button>
        <button id=\"stopBtn\" disabled>Stop</button>
        <button id=\"resetBtn\">Reset</button>
      </div>
      <div class=\"status\" id=\"status\">Idle.</div>
      <div class=\"panel\">
        <span class=\"label\">Transcript</span>
        <div id=\"transcript\">—</div>
        <span class=\"label\" style=\"margin-top:12px;\">Chat Log</span>
        <div id=\"chatLog\" class=\"log\"></div>
        <span class=\"label\" style=\"margin-top:12px;\">Your Input</span>
        <audio id=\"audioInput\" controls></audio>
        <span class=\"label\" style=\"margin-top:12px;\">Reply Audio</span>
        <audio id=\"audioReply\" controls></audio>
      </div>
    </div>

    <script>
      const recordBtn = document.getElementById("recordBtn");
      const stopBtn = document.getElementById("stopBtn");
      const resetBtn = document.getElementById("resetBtn");
      const statusEl = document.getElementById("status");
      const transcriptEl = document.getElementById("transcript");
      const chatLog = document.getElementById("chatLog");
      const audioInput = document.getElementById("audioInput");
      const audioReply = document.getElementById("audioReply");

      let mediaRecorder = null;
      let chunks = [];
      let stream = null;

      function setStatus(text) {
        statusEl.textContent = text;
      }

      function resetButtons(isRecording) {
        recordBtn.disabled = isRecording;
        stopBtn.disabled = !isRecording;
      }

      function addLogEntry(role, message) {
        const entry = document.createElement("div");
        entry.className = "log-entry";
        const roleSpan = document.createElement("span");
        roleSpan.className = "log-role";
        roleSpan.textContent = role + ": ";
        const textSpan = document.createElement("span");
        textSpan.textContent = message;
        entry.appendChild(roleSpan);
        entry.appendChild(textSpan);
        chatLog.appendChild(entry);
        chatLog.scrollTop = chatLog.scrollHeight;
      }

      function floatTo16BitPCM(view, offset, input) {
        for (let i = 0; i < input.length; i++, offset += 2) {
          let s = Math.max(-1, Math.min(1, input[i]));
          view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
        }
      }

      function writeString(view, offset, string) {
        for (let i = 0; i < string.length; i++) {
          view.setUint8(offset + i, string.charCodeAt(i));
        }
      }

      function audioBufferToWav(buffer) {
        const sampleRate = buffer.sampleRate;
        const numChannels = 1;
        const length = buffer.length;
        const monoData = new Float32Array(length);
        for (let ch = 0; ch < buffer.numberOfChannels; ch++) {
          const channelData = buffer.getChannelData(ch);
          for (let i = 0; i < length; i++) {
            monoData[i] += channelData[i] / buffer.numberOfChannels;
          }
        }

        const bufferLength = 44 + monoData.length * 2;
        const arrayBuffer = new ArrayBuffer(bufferLength);
        const view = new DataView(arrayBuffer);

        writeString(view, 0, "RIFF");
        view.setUint32(4, 36 + monoData.length * 2, true);
        writeString(view, 8, "WAVE");
        writeString(view, 12, "fmt ");
        view.setUint32(16, 16, true);
        view.setUint16(20, 1, true);
        view.setUint16(22, numChannels, true);
        view.setUint32(24, sampleRate, true);
        view.setUint32(28, sampleRate * numChannels * 2, true);
        view.setUint16(32, numChannels * 2, true);
        view.setUint16(34, 16, true);
        writeString(view, 36, "data");
        view.setUint32(40, monoData.length * 2, true);
        floatTo16BitPCM(view, 44, monoData);

        return new Blob([view], { type: "audio/wav" });
      }

      function getSessionId() {
        let sessionId = localStorage.getItem("atta_session_id");
        if (!sessionId) {
          sessionId = crypto.randomUUID();
          localStorage.setItem("atta_session_id", sessionId);
        }
        return sessionId;
      }

      async function sendAudio(wavBlob) {
        setStatus("Uploading and processing...");
        transcriptEl.textContent = "—";
        audioReply.removeAttribute("src");

        const formData = new FormData();
        formData.append("file", new File([wavBlob], "recording.wav", { type: "audio/wav" }));

        const response = await fetch("/voice_chat/", {
          method: "POST",
          headers: {
            "X-Session-Id": getSessionId()
          },
          body: formData
        });

        if (!response.ok) {
          const detail = await response.text();
          throw new Error(detail || "Request failed");
        }

        const data = await response.json();
        transcriptEl.textContent = data.transcription || "(no transcription)";
        if (data.transcription) {
          addLogEntry("User", data.transcription);
        }
        if (data.response) {
          addLogEntry("LLM", data.response);
        }

        if (data.audio) {
          const binary = atob(data.audio);
          const bytes = new Uint8Array(binary.length);
          for (let i = 0; i < binary.length; i++) {
            bytes[i] = binary.charCodeAt(i);
          }
          const audioBlob = new Blob([bytes], { type: "audio/wav" });
          const url = URL.createObjectURL(audioBlob);
          audioReply.src = url;
          await audioReply.play();
          if (data.history_size) {
            setStatus("Done. History messages: " + data.history_size + ".");
          } else {
            setStatus("Done.");
          }
        } else {
          setStatus("Done, but no audio returned.");
        }
      }

      recordBtn.addEventListener("click", async () => {
        try {
          audioInput.removeAttribute("src");
          audioInput.load();
          stream = await navigator.mediaDevices.getUserMedia({ audio: true });
          chunks = [];
          mediaRecorder = new MediaRecorder(stream);
          mediaRecorder.ondataavailable = (event) => {
            if (event.data && event.data.size > 0) {
              chunks.push(event.data);
            }
          };

          mediaRecorder.onstop = async () => {
            try {
              const blob = new Blob(chunks);
              const arrayBuffer = await blob.arrayBuffer();
              const audioContext = new (window.AudioContext || window.webkitAudioContext)();
              const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
              const wavBlob = audioBufferToWav(audioBuffer);
              const inputUrl = URL.createObjectURL(wavBlob);
              audioInput.src = inputUrl;
              await sendAudio(wavBlob);
            } catch (err) {
              setStatus("Error: " + err.message);
            } finally {
              if (stream) {
                stream.getTracks().forEach((track) => track.stop());
              }
              resetButtons(false);
            }
          };

          mediaRecorder.start();
          setStatus("Recording...");
          resetButtons(true);
        } catch (err) {
          setStatus("Microphone error: " + err.message);
        }
      });

      stopBtn.addEventListener("click", () => {
        if (mediaRecorder && mediaRecorder.state !== "inactive") {
          mediaRecorder.stop();
          setStatus("Stopping...");
        }
      });

      resetBtn.addEventListener("click", async () => {
        try {
          const response = await fetch("/reset_chat/", {
            method: "POST",
            headers: {
              "X-Session-Id": getSessionId()
            }
          });
          if (!response.ok) {
            const detail = await response.text();
            throw new Error(detail || "Reset failed");
          }
          transcriptEl.textContent = "—";
          audioReply.removeAttribute("src");
          audioInput.removeAttribute("src");
          chatLog.textContent = "";
          setStatus("Conversation reset.");
        } catch (err) {
          setStatus("Reset error: " + err.message);
        }
      });
    </script>
  </body>
</html>
"""


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/transcribe/")
async def handle_transcription(file: UploadFile = File(...)):
    """Transcribe an audio file."""
    if not file.filename.lower().endswith(".wav"):
        raise HTTPException(400, "Only WAV files are allowed")

    file_ext = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_path = tmp_file.name

    try:
        transcription = transcribe_audio_file(tmp_path)
    except Exception as e:
        raise HTTPException(500, f"Processing error: {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return {"transcription": transcription}


@app.post("/voice_chat/")
async def voice_chat(
    file: UploadFile = File(...),
    session_id: str | None = Header(None, alias="X-Session-Id"),
):
    """Handle WAV upload, transcribe, chat, and return TTS audio."""
    if not file.filename.lower().endswith(".wav"):
        raise HTTPException(400, "Only WAV files are allowed")

    recordings_dir = _get_recordings_dir()
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    safe_name = f"recording_{timestamp}_{uuid4().hex[:8]}.wav"
    saved_path = recordings_dir / safe_name

    content = await file.read()
    with open(saved_path, "wb") as f:
        f.write(content)

    try:
        session_key = _get_session_id(session_id)
        transcription = transcribe_audio_file(str(saved_path))
        _append_message(session_key, "user", transcription)
        message_history = _get_messages(session_key)
        messages_for_llm = _ensure_polish_system_prompt(message_history)
        response_text = chat_with_llm_messages(messages_for_llm)
        _append_message(session_key, "assistant", response_text)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_path = tmp_file.name

        try:
            from src.services.tts import generate_speech

            await generate_speech(response_text, tmp_path)
            with open(tmp_path, "rb") as f:
                audio_data = f.read()
            audio_base64 = base64.b64encode(audio_data).decode("utf-8")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        return {
            "transcription": transcription,
            "response": response_text,
            "audio": audio_base64,
            "saved_file": safe_name,
            "history_size": len(messages_for_llm),
        }
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Error connecting to LLM: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@app.post("/chat_llm/")
async def chat(
    text: str = Body(..., embed=True),
    session_id: str | None = Header(None, alias="X-Session-Id"),
):
    """Chat with LLM."""
    try:
        session_key = _get_session_id(session_id)
        _append_message(session_key, "user", text)
        message_history = _get_messages(session_key)
        messages_for_llm = _ensure_polish_system_prompt(message_history)
        response_text = chat_with_llm_messages(messages_for_llm)
        _append_message(session_key, "assistant", response_text)
        return {"response": response_text, "history_size": len(messages_for_llm)}
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Error connecting to LLM: {str(e)}")
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reset_chat/")
async def reset_chat(session_id: str | None = Header(None, alias="X-Session-Id")):
    """Reset the conversation history for a session."""
    session_key = _get_session_id(session_id)
    _reset_messages(session_key)
    return {"status": "ok"}




@app.post("/tts/")
async def tts_endpoint(
    text: str = Form(...),
    lang: str = Form("pl"),
    format: str = Form("wav"),
):
    """Text-to-speech endpoint using edge-tts."""
    from src.services.tts import generate_speech

    if not text or not text.strip():
        raise HTTPException(400, "Text cannot be empty")

    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{format}") as tmp_file:
        tmp_path = tmp_file.name

    try:
        await generate_speech(text, tmp_path)

        if format == "wav":
            audio = AudioSegment.from_file(tmp_path)
            sample_rate = audio.frame_rate
            channels = audio.channels
            sample_width = audio.sample_width
        else:
            sample_rate = 24000
            channels = 1
            sample_width = 2

        with open(tmp_path, "rb") as f:
            audio_data = f.read()

        audio_base64 = base64.b64encode(audio_data).decode("utf-8")

        return JSONResponse(
            content={
                "audio": audio_base64,
                "sample_rate": sample_rate,
                "channels": channels,
                "sample_width": sample_width,
                "format": format,
            }
        )
    except Exception as e:
        raise HTTPException(500, f"TTS generation failed: {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
