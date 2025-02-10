import requests
import json

class ATTAPI:
    def __init__(self, url):
        self.url = url

    def transcribe(self, file_path: str):
        with open(file_path, "rb") as audio_file:
            files = {"file": audio_file}
            response = requests.post(self.url + "transcribe/", files=files)
        if response.status_code == 200:
            data = response.json()
            data.get("chat_response")
            return data
        else:
            print(f"Error: {response.status_code}")
            print(response.text)

    def llm(self, text: str):
        payload = {"text": text}  # Structure data properly
        
        try:
            response = requests.post(self.url + "chat_llm/", json=payload)
            response.raise_for_status()  # Raises an HTTPError for bad responses (4xx, 5xx)
            return response.text
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            return None

# def tts(text: str):
#     response = requests.post(url+"tts/", data=text)
#     if response.status_code == 200:
#         print(response.text)
#         return response.text
#     else:
#         print(f"Error {response.status_code}: {response.text}")