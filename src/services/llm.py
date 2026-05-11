import httpx

from src.app.core.config import settings


def chat_with_llm(text: str) -> str:
    """Send text to LLM via Ollama and return the response."""
    return chat_with_llm_messages([{"role": "user", "content": text}])


def chat_with_llm_messages(messages: list[dict[str, str]]) -> str:
    """Send a message history to LLM via Ollama and return the response."""
    url = f"{settings.ollama_base_url}/api/chat"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": settings.ollama_model,
        "messages": messages,
        "stream": False,
        "max_length": 128,
    }
    with httpx.Client(timeout=60.0) as client:
        response = client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
    if "message" in result and "content" in result["message"]:
        return result["message"]["content"]
    raise ValueError("Invalid response format from LLM")
