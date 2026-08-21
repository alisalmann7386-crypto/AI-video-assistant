import os
import re
import tempfile
import yt_dlp
import streamlit as st
from pydub import AudioSegment


def is_youtube_url(url: str) -> bool:
    """Checks if a string is a valid YouTube URL."""
    youtube_regex = r'(https?://)?(www\.)?(youtube|youtu|youtube-nocookie)\.(com|be)/(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})'
    return bool(re.match(youtube_regex, url))


def split_audio_into_chunks(audio_path: str, chunk_length_ms: int = 10 * 60 * 1000) -> list[str]:
    """Splits an audio file into smaller chunks (default 10 minutes) for API compatibility."""
    audio = AudioSegment.from_file(audio_path)
    
    if len(audio) <= chunk_length_ms:
        return [audio_path]

    base_dir = os.path.dirname(audio_path)
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    chunks = []

    for i, chunk in enumerate(audio[::chunk_length_ms]):
        chunk_path = os.path.join(base_dir, f"{base_name}_chunk_{i}.mp3")
        chunk.export(chunk_path, format="mp3")
        chunks.append(chunk_path)

    return chunks


def get_cookies_filepath() -> str | None:
    """Reads YouTube cookies from Streamlit secrets and writes them to a temporary file."""
    try:
        if "YOUTUBE_COOKIES" in st.secrets and st.secrets["YOUTUBE_COOKIES"].strip():
            temp_cookie_file = tempfile.NamedTemporaryFile(
                mode="w+", delete=False, suffix=".txt"
            )
            temp_cookie_file.write(st.secrets["YOUTUBE_COOKIES"])
            temp_cookie_file.close()
            return temp_cookie_file.name
    except Exception as e:
        print(f"Warning: Could not read cookies from Streamlit secrets: {e}")
    return None


def download_youtube_audio(url: str) -> tuple[list[str], dict]:
    """Downloads audio from YouTube bypassing Cloud IP blocks using modern client extractors."""
    output_dir = tempfile.mkdtemp()
    out_template = os.path.join(output_dir, "%(id)s.%(ext)s")
    
    cookie_path = get_cookies_filepath()

    ydl_opts = {
        "format": "ba/b",
        "outtmpl": out_template,
        # Force clients that bypass Cloud IP 403 blocks (Android Creator & TV Embedded)
        "extractor_args": {
            "youtube": {
                "player_client": ["android_creator", "tv_embedded", "ios"],
                "skip": ["dash", "hls"]
            }
        },
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }],
        "quiet": True,
        "no_warnings": True,
        "nocheckcertificate": True,
    }

    if cookie_path:
        ydl_opts["cookiefile"] = cookie_path

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info)
            audio_filepath = os.path.splitext(filename)[0] + ".mp3"

            duration_sec = info.get("duration", 0) or 0
            metadata = {
                "title": info.get("title", "YouTube Video"),
                "channel": info.get("uploader", "Unknown Channel"),
                "duration": f"{duration_sec // 60}m {duration_sec % 60}s",
                "thumbnail": info.get("thumbnail", ""),
                "url": url
            }

        chunks = split_audio_into_chunks(audio_filepath)
        return chunks, metadata

    finally:
        if cookie_path and os.path.exists(cookie_path):
            os.remove(cookie_path)


def process_local_file(file_path: str) -> tuple[list[str], dict]:
    """Processes locally uploaded audio/video files into MP3 format and chunks them."""
    file_name = os.path.basename(file_path)
    base_name, ext = os.path.splitext(file_name)
    ext = ext.lower().replace(".", "")

    output_path = os.path.join(os.path.dirname(file_path), f"{base_name}_converted.mp3")

    audio = AudioSegment.from_file(file_path, format=ext if ext != "mkv" else "matroska")
    audio.export(output_path, format="mp3")

    duration_sec = int(len(audio) / 1000)
    metadata = {
        "title": base_name,
        "channel": "Local Upload",
        "duration": f"{duration_sec // 60}m {duration_sec % 60}s",
        "thumbnail": None,
        "url": None
    }

    chunks = split_audio_into_chunks(output_path)
    return chunks, metadata


def process_input(source: str) -> tuple[list[str], dict]:
    """Main routing function."""
    if is_youtube_url(source):
        return download_youtube_audio(source)
    else:
        return process_local_file(source)
