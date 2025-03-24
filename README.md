## Api Audio to text to audio
ATA is a voice chatbot. It gets audio and returns proceed audio by LLM.
For now everything is running on Fast API.

The program is currently using my tunned version of whisper-base model "marcsixtysix/whisper-base-pl" to speech recognition.
You can see the model in my github repository:
https://github.com/mryt66/Speech-recognition-pl

Output of speech recognition model is proceed by Language tool, that formats the string that is later sent to Large Language Model "bielik-11b-v2.3-instruct:Q6_K" using Ollama.
After getting the response of LLM it is sent to Zonos model that creates a voice output.

<p align="center">
  <img src="https://github.com/user-attachments/assets/e5fff0b9-65b6-46dc-9d97-237ceb94f53d" />
  <br />
  Example of program usage
</p>
