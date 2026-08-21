import os
import re
import tempfile
import yt_dlp
from pydub import AudioSegment


def is_youtube_url(url: str) -> bool:
    """Checks if a string is a valid YouTube URL."""
    youtube_regex = r'(https?://)?(www\.)?(youtube|youtu|youtube-nocookie)\.(com|be)/(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})'
    return bool(re.match(youtube_regex, url))


def download_youtube_audio(url: str) -> tuple[str, dict]:
    """Downloads audio from YouTube using yt-dlp with mobile client overrides

    to prevent HTTP 403 Forbidden errors on cloud servers.
    """
    output_dir = tempfile.mkdtemp()
    out_template = os.path.join(output_dir, "%(id)s.%(ext)s")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": out_template,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }],
        # CRITICAL 403 FIX: Force yt-dlp to mimic mobile clients
        "extractor_args": {
            "youtube": {
                "player_client": ["android", "ios"]
            }
        },
        "http_headers": {
            "User-Agent": (
                "Mozilla/5.0 (iPhone; CPU iPhone OS 16_6 like Mac OS X) "
                "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.6 Mobile/15E148 Safari/604.1"
            )
        },
        "quiet": True,
        "no_warnings": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)
        
        # Audio gets converted to .mp3 by postprocessor
        audio_filepath = os.path.splitext(filename)[0] + ".mp3"

        metadata = {
            "title": info.get("title", "YouTube Video"),
            "channel": info.get("uploader", "Unknown Channel"),
            "duration": f"{info.get('duration', 0) // 60}m {info.get('duration', 0) % 60}s",
            "thumbnail": info.get("thumbnail", ""),
            "url": url
        }

    return audio_filepath, metadata


def process_local_file(file_path: str) -> tuple[str, dict]:
    """Processes locally uploaded audio or video files into MP3 format."""
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

    return output_path, metadata


def process_input(source: str) -> tuple[str, dict]:
    """Main routing function that takes either a YouTube URL or local file path

    and returns a single audio file path along with video/audio metadata.
    """
    if is_youtube_url(source):
        return download_youtube_audio(source)
    else:
        return process_local_file(source)
