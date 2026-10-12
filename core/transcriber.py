import os
from groq import Groq
from dotenv import load_dotenv
load_dotenv()

def get_groq_client(api_key: str = None) -> Groq:
    key = api_key or os.getenv("GROQ_API_KEY")
    if not key:
        raise ValueError("GROQ_API_KEY is not set. Please provide it in .env or via settings.")
    return Groq(api_key=key)

client = None
try:
    if os.getenv("GROQ_API_KEY"):
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
except Exception:
    client = None

WHISPER_MODEL = "whisper-large-v3"

def transcribe_chunk(chunk_path: str, groq_client: Groq = None, model: str = WHISPER_MODEL) -> str:
    """Reads an audio chunk and translates it directly to English."""
    c = groq_client or client or get_groq_client()
    with open(chunk_path, "rb") as audio_file:
        response = c.audio.translations.create(
            file=(os.path.basename(chunk_path), audio_file.read()),
            model=model,
            response_format="text",
        )

    return response.strip() if isinstance(response, str) else response.text.strip()

def transcribe_all(chunks: list, lan: str = 'english', progress_callback = None, groq_client: Groq = None) -> str:
    full_transcript = ""
    total = len(chunks)
    c = groq_client or client or get_groq_client()
    for i, chunk in enumerate(chunks):
        if progress_callback:
            progress_callback(i + 1, total)
        else:
            print(f"Transcribing chunk {i+1}/{total}")
        text = transcribe_chunk(chunk, groq_client=c)
        full_transcript += text + " "

    print("Transcription completed")
    return full_transcript.strip()
