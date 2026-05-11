from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "hf.co/marcsixtysix/gemma-3-1b-it-pl-polqa"
    whisper_model: str = "marcsixtysix/whisper-base-pl"
    language: str = "pl"
    tts_voice: str = "pl-PL-MarekNeural"
    api_port: int = 8000

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
