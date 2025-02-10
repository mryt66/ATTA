import language_tool_python
from transformers import pipeline

# pipe1 = pipeline(model="marcsixtysix/whisper-base-pl")
pipe1 = pipe = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3")
tool1 = language_tool_python.LanguageTool('pl')

def correct_polish_text(text):
    max_iterations = 3 
    for _ in range(max_iterations):
        matches = tool1.check(text)
        if not matches:
            break
        corrected_parts = []
        last_pos = 0
        for match in matches:
            start = match.offset
            end = start + match.errorLength
            corrected_parts.append(text[last_pos:start])
            if match.replacements:
                corrected_parts.append(match.replacements[0])
            else:
                corrected_parts.append(text[start:end])
            last_pos = end
        corrected_parts.append(text[last_pos:])
        text = ''.join(corrected_parts)
    return text

def transcribe_audio_file(file_path: str) -> str:
    """Transcribe audio file and return corrected text"""
    result = pipe1(file_path, generate_kwargs={"language": "polish"})
    return correct_polish_text(result["text"])
