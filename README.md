## Api Audio to text to audio
ATTA is a voice chatbot. It gets audio and returns proceed audio by LLM.
For now everything is running on Fast API.

The program is currently using my tunned version of whisper-base model "marcsixtysix/whisper-base-pl" to speech recognition.
You can see the model in my github repository:
https://github.com/mryt66/Speech-recognition-pl

The output from the speech recognition model is processed by a language tool that formats the text before sending it to the large language model "marcsixtysix/gemma-3-4b-it-pl-polqa" via Ollama.
For this project, I have fine-tuned the Gemma-3-1B-IT model to function as a Polish-language Q&A system.
You can find it here: https://huggingface.co/marcsixtysix/gemma-3-1b-it-pl-polqa-GGUF

Once a response is generated by the LLM, it is passed to the edge_tts, which converts the text into voice output.

<p align="center">
  <img src="https://github.com/user-attachments/assets/88473e61-baa6-48c9-8239-8360c747c310" />
  <br />
  Example of program usage
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/ee4b560d-908d-43f6-8121-499d65215f54" />
  <br />
  Api's endpoints
</p>

