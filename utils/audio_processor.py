import os
import tempfile
from pydub import AudioSegment
import streamlit as st
import yt_dlp

DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)


def get_cookie_file_path() -> str | None:
    """Writes cookies from Streamlit secrets to a temp file if available."""
    if "YOUTUBE_COOKIES" in st.secrets:
        temp_cookie = tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".txt"
        )
        temp_cookie.write(st.secrets["YOUTUBE_COOKIES"])
        temp_cookie.flush()
        return temp_cookie.name
    return None


def download_yt_audio(url: str) -> str:
    output_path = os.path.join(DOWNLOAD_DIR, "%(title)s.%(ext)s")
    cookie_path = get_cookie_file_path()

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        # 1. Enable Node.js installed via packages.txt
        "js_runtimes": {"node": {}},
        # 2. Prefer clients that avoid heavy web challenges
        "extractor_args": {
            "youtube": {
                "player_client": ["ios", "android"],
            }
        },
        "http_headers": {
            "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 17_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Mobile/15E148 Safari/604.1",
            "Accept-Language": "en-US,en;q=0.9",
        },
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "128",
            }
        ],
        "quiet": False,
        "no_warnings": False,
    }

    if cookie_path:
        ydl_opts["cookiefile"] = cookie_path

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            base, _ = os.path.splitext(ydl.prepare_filename(info))
            filename = f"{base}.mp3"
        return filename
    finally:
        # Clean up temporary cookie file
        if cookie_path and os.path.exists(cookie_path):
            os.remove(cookie_path)


def convert_to_wav(input_path: str) -> str:
    """Convert any local file to standard audio format."""
    output_path = os.path.splitext(input_path)[0] + "_converted.mp3"
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1).set_frame_rate(16000)
    audio.export(output_path, format="mp3", bitrate="64k")
    return output_path


def chunk_audio(audio_path: str, chunk_minutes: int = 5) -> list:
    audio = AudioSegment.from_file(audio_path)
    audio = audio.set_channels(1).set_frame_rate(16000)

    chunk_ms = chunk_minutes * 60 * 1000
    chunks = []
    base_name = os.path.splitext(audio_path)[0]

    for i, start in enumerate(range(0, len(audio), chunk_ms)):
        chunk = audio[start : start + chunk_ms]
        chunk_path = f"{base_name}_chunk_{i}.mp3"
        chunk.export(chunk_path, format="mp3", bitrate="64k")
        chunks.append(chunk_path)

    return chunks


def get_audio_path(source: str) -> str:
    if source.startswith("http://") or source.startswith("https://"):
        return download_yt_audio(source)
    return convert_to_wav(source)


def process_input(source: str, chunk_minutes: int = 5) -> list:
    if source.startswith("http://") or source.startswith("https://"):
        print("Detected YouTube URL. Downloading audio...")
        audio_path = download_yt_audio(source)
    else:
        print("Detected local file. Converting audio...")
        audio_path = convert_to_wav(source)

    print("Chunking audio...")
    chunks = chunk_audio(audio_path, chunk_minutes=chunk_minutes)
    print(f"Audio ready - {len(chunks)} chunks created.")
    return chunks