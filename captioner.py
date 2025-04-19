import whisper
import gradio as gr
from pytube import YouTube
import os
import download

model = whisper.load_model("base")  # can also be "small", "medium", etc.


def transcribe_audio(filepath):
    print(filepath)
    result = model.transcribe(filepath, fp16=False)
    print(result["text"])
    return result["text"]


def generate_captions(youtube_url):
    try:
        filepath = download.download_audio(youtube_url)
        transcript = transcribe_audio(filepath)
        os.remove(filepath)
        return transcript
    except Exception as e:
        return f"⚠️ Error: {str(e)}"


gr.Interface(
    fn=generate_captions,
    inputs=gr.Textbox(label="YouTube URL"),
    outputs=gr.Textbox(label="Transcription"),
    title="Captioner",
    description="Paste a YouTube link to generate AI-generated captions using OpenAI Whisper.",
).launch()
