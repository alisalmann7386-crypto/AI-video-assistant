import os
import re
import tempfile
import yt_dlp
from pydub import AudioSegment


def is_youtube_url(url: str) -> bool:
    """Checks if a string is a valid YouTube URL."""
    youtube_regex = r'(https?://)?(www\.)?(youtube|youtu|youtube-nocookie)\.(com|be)/(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})'
    return bool(re.match(youtube_regex, url))


def split_audio_into_chunks(audio_path: str, chunk_length_ms: int = 10 * 60 * 1000) -> list[str]:
    """Splits an audio file into smaller chunks (default 10 minutes) to fit API payload limits."""
    audio = AudioSegment.from_file(audio_path)
    chunks = []
    
    if len(audio) <= chunk_length_ms:
        return [audio_path]

    base_dir = os.path.dirname(audio_path)
    base_name = os.path.splitext(os.path.basename(audio_path))[0]

    for i, chunk in enumerate(audio[::chunk_length_ms]):
        chunk_path = os.path.join(base_dir, f"{base_name}_chunk_{i}.mp3")
        chunk.export(chunk_path, format="mp3")
        chunks.append(chunk_path)

    return chunks


def download_youtube_audio(url: str) -> tuple[list[str], dict]:
    """Downloads audio from YouTube using yt-dlp with updated multi-client fallbacks

    to bypass HTTP 403 Forbidden errors on cloud host servers.
    """
    output_dir = tempfile.mkdtemp()
    out_template = os.path.join(output_dir, "%(id)s.%(ext)s")

    ydl_opts = {
        "format": "ba/ba*",
        "outtmpl": out_template,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }],
        # ── Updated 403 Fix: Multi-Client Fallback Strategy ────────────────────
        "nocheckcertificate": True,
        "ignoreerrors": False,
        "logtostderr": False,
        "quiet": True,
        "no_warnings": True,
        "source_address": "0.0.0.0",  # Force IPv4 connection
        "extractor_args": {
            "youtube": {
                "player_client": ["android_vr", "web_embedded", "android", "ios", "tv"],
                "player_skip": ["webpage", "configs"]
            }
        },
        "http_headers": {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/125.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)
        
        # Audio gets converted to .mp3 by postprocessor
        audio_filepath = os.path.splitext(filename)[0] + ".mp3"

        duration_sec = info.get("duration", 0)
        metadata = {
            "title": info.get("title", "YouTube Video"),
            "channel": info.get("uploader", "Unknown Channel"),
            "duration": f"{duration_sec // 60}m {duration_sec % 60}s",
            "thumbnail": info.get("thumbnail", ""),
            "url": url
        }

    chunks = split_audio_into_chunks(audio_filepath)
    return chunks, metadata


def process_local_file(file_path: str) -> tuple[list[str], dict]:
    """Processes locally uploaded audio or video files into MP3 format and chunks them."""
    file_name = os.path.basename(file_path)
    base_name, ext = os.path.splitext(file_name)
    ext = ext.lower().replace(".", "")

    output_path = os.path.join(os.path.dirname(file_path), f"{base_name}_converted.mp3")

    # Extract audio using pydub
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
    """Main routing function that takes either a YouTube URL or local file path

    and returns a list of chunked audio file paths along with video/audio metadata.
    """
    if is_youtube_url(source):
        return download_youtube_audio(source)
    else:
        return process_local_file(source)
