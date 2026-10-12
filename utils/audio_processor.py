import os
from pydub import AudioSegment
import yt_dlp

DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

def download_yt_audio(url: str) -> str:
    output_path = os.path.join(DOWNLOAD_DIR, "%(title)s.%(ext)s")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        # 1. Use mobile client to bypass datacenter 403 blocks & JS challenge
        "extractor_args": {
            "youtube": {
                "player_client": ["android", "web"]
            }
        },
        # 2. Add realistic user-agent header
        "http_headers": {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
        },
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "128",
            }
        ],
        "quiet": True,
        "no_warnings": False,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        base, _ = os.path.splitext(ydl.prepare_filename(info))
        filename = f"{base}.mp3"
    return filename

def convert_to_wav(input_path: str) -> str:
    """Convert any local file to standard audio format."""
    output_path = os.path.splitext(input_path)[0] + "_converted.mp3"
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1).set_frame_rate(16000)
    audio.export(output_path, format="mp3", bitrate="64k")
    return output_path


def chunk_audio(audio_path: str, chunk_minutes: int = 5) -> list:
    # 2. Use AudioSegment.from_file (handles both wav and mp3)
    audio = AudioSegment.from_file(audio_path)

    # 3. Downsample to mono and 16000 Hz (native to Whisper, cuts size)
    audio = audio.set_channels(1).set_frame_rate(16000)

    chunk_ms = chunk_minutes * 60 * 1000
    chunks = []
    base_name = os.path.splitext(audio_path)[0]

    for i, start in enumerate(range(0, len(audio), chunk_ms)):
        chunk = audio[start : start + chunk_ms]

        # 4. Export as compressed MP3 under ~2.5 MB instead of a 55 MB uncompressed WAV
        chunk_path = f"{base_name}_chunk_{i}.mp3"
        chunk.export(chunk_path, format="mp3", bitrate="64k")
        chunks.append(chunk_path)

    return chunks


def get_audio_path(source: str) -> str:
    if source.startswith("http://") or source.startswith("https://"):
        return download_yt_audio(source)
    else:
        return convert_to_wav(source)


def process_input(source: str, chunk_minutes: int = 5) -> list:
    if source.startswith("http://") or source.startswith("https://"):
        print("detected Youtube URL. Downloading audio...")
        audio_path = download_yt_audio(source)
    else:
        print("detected local file. Converting audio...")
        audio_path = convert_to_wav(source)

    print("Chunking audio")
    chunks = chunk_audio(audio_path, chunk_minutes=chunk_minutes)
    print(f"Audio ready - {len(chunks)} chunks created")
    return chunks

#https://www.youtube.com/watch?v=lZNv3ORKs98