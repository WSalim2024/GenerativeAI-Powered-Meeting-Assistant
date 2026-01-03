import gradio as gr
from faster_whisper import WhisperModel
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Initialize Local LLM (Ollama) - No subscription required
llm = OllamaLLM(model="llama3")

# 2. Modern 2026 Prompt Template
template = """
<s><<SYS>>
You are a high-level executive assistant. 
List the key points with details from the following meeting context.
Keep the tone professional and action-oriented.
<</SYS>>

[INST] The context : {context} [/INST]
"""

prompt = PromptTemplate.from_template(template)

# 3. Create the Chain using LCEL (Standard for 2026)
# This replaces the old LLMChain and is much faster
meeting_chain = prompt | llm | StrOutputParser()

# 4. Initialize Local Transcription Model (Faster-Whisper)
# Optimized to run 4x-10x faster than standard Whisper on your CPU
stt_model = WhisperModel("tiny", device="cpu", compute_type="int8")


def transcript_audio(audio_file):
    # STEP A: Transcribe locally
    segments, _ = stt_model.transcribe(audio_file, beam_size=5)
    transcript_txt = " ".join([s.text for s in segments])

    # STEP B: Analyze with local Llama 3
    # We use .invoke() instead of .run() in 2026
    result = meeting_chain.invoke({"context": transcript_txt})

    return result


# 5. Gradio Interface - Simple and Precise
iface = gr.Interface(
    fn=transcript_audio,
    inputs=gr.Audio(type="filepath", label="Upload Meeting Audio"),
    outputs=gr.Textbox(label="Executive Summary & Key Points"),
    title="Audio Transcription App (Local & Free)",
    description="Securely transcribe and summarize your brainstorming sessions."
)

if __name__ == "__main__":
    # Note: Use share=True if you want to access this via a public link temporarily
    iface.launch(server_name="0.0.0.0", server_port=7860)