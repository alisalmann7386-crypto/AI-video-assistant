import os
import requests
from typing import List, Dict, Any, Optional
from pydub import AudioSegment
from groq import Groq


def format_timestamp(seconds: float) -> str:
    """Converts seconds into HH:MM:SS or MM:SS format."""
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def get_audio_duration(file_path: str) -> float:
    """Gets duration of an audio file in seconds."""
    try:
        audio = AudioSegment.from_file(file_path)
        return len(audio) / 1000.0
    except Exception:
        return 0.0


# ── Groq API Engine (Fast Cloud Whisper) ──────────────────────────────────────
def transcribe_chunk_groq(
    chunk_path: str,
    api_key: str,
    time_offset: float = 0.0
) -> tuple[str, list[dict]]:
    """Transcribes audio chunk via Groq Cloud Whisper API."""
    client = Groq(api_key=api_key)
    
    with open(chunk_path, "rb") as file:
        transcription = client.audio.transcriptions.create(
            file=(os.path.basename(chunk_path), file.read()),
            model="whisper-large-v3",
            response_format="verbose_json"
        )

    chunk_text = transcription.text.strip()
    segments = []

    # Parse segments with timestamp offsets
    raw_segments = getattr(transcription, "segments", [])
    if raw_segments:
        for seg in raw_segments:
            # Handle dictionary or object response from SDK
            start_val = seg.get("start", 0.0) if isinstance(seg, dict) else getattr(seg, "start", 0.0)
            end_val = seg.get("end", 0.0) if isinstance(seg, dict) else getattr(seg, "end", 0.0)
            text_val = seg.get("text", "") if isinstance(seg, dict) else getattr(seg, "text", "")

            start_sec = start_val + time_offset
            end_sec = end_val + time_offset

            segments.append({
                "start": format_timestamp(start_sec),
                "end": format_timestamp(end_sec),
                "start_raw": start_sec,
                "end_raw": end_sec,
                "text": text_val.strip()
            })
    else:
        duration = get_audio_duration(chunk_path)
        segments.append({
            "start": format_timestamp(time_offset),
            "end": format_timestamp(time_offset + duration),
            "start_raw": time_offset,
            "end_raw": time_offset + duration,
            "text": chunk_text
        })

    return chunk_text, segments


# ── Sarvam AI Engine (Indic Languages) ─────────────────────────────────────────
def transcribe_chunk_sarvam(
    chunk_path: str,
    api_key: str,
    time_offset: float = 0.0,
    mode: str = "transcribe"
) -> tuple[str, list[dict]]:
    """Transcribes audio chunk using Sarvam AI REST API."""
    url = "https://api.sarvam.ai/speech-to-text"
    headers = {"api-subscription-key": api_key}
    
    payload = {
        "model": "saaras:v3",
        "mode": mode,
        "language_code": "unknown",
        "with_timestamps": "true"
    }
    
    with open(chunk_path, "rb") as audio_file:
        files = {"file": (os.path.basename(chunk_path), audio_file, "audio/wav")}
        response = requests.post(url, headers=headers, data=payload, files=files)
        
    if response.status_code != 200:
        raise RuntimeError(f"Sarvam API error ({response.status_code}): {response.text}")
        
    data = response.json()
    chunk_text = data.get("transcript", "").strip()
    segments = []
    
    timestamps_obj = data.get("timestamps")
    if timestamps_obj and isinstance(timestamps_obj, dict):
        words_or_chunks = timestamps_obj.get("words", [])
        for entry in words_or_chunks:
            start_sec = entry.get("start_time_seconds", 0.0) + time_offset
            end_sec = entry.get("end_time_seconds", 0.0) + time_offset
            
            segments.append({
                "start": format_timestamp(start_sec),
                "end": format_timestamp(end_sec),
                "start_raw": start_sec,
                "end_raw": end_sec,
                "text": entry.get("word", "").strip() or entry.get("transcript", "").strip()
            })
    else:
        chunk_duration = get_audio_duration(chunk_path)
        segments.append({
            "start": format_timestamp(time_offset),
            "end": format_timestamp(time_offset + chunk_duration),
            "start_raw": time_offset,
            "end_raw": time_offset + chunk_duration,
            "text": chunk_text
        })
        
    return chunk_text, segments


# ── Orchestrator Engine ───────────────────────────────────────────────────────
def process_transcription(
    chunks: List[str],
    provider: str = "groq",
    translate_to_english: bool = False
) -> Dict[str, Any]:
    """Orchestrates API-based transcriptions."""
    full_text_list = []
    all_segments = []
    cumulative_offset = 0.0

    groq_key = os.getenv("GROQ_API_KEY")
    sarvam_key = os.getenv("SARVAM_API_KEY")

    for chunk_path in chunks:
        if not os.path.exists(chunk_path):
            continue

        if provider == "sarvam":
            if not sarvam_key:
                raise ValueError("SARVAM_API_KEY missing. Set it in .env or Streamlit Secrets.")
            
            mode = "translate" if translate_to_english else "transcribe"
            chunk_text, segments = transcribe_chunk_sarvam(
                chunk_path=chunk_path,
                api_key=sarvam_key,
                time_offset=cumulative_offset,
                mode=mode
            )
        else:
            if not groq_key:
                raise ValueError("GROQ_API_KEY missing. Set it in .env or Streamlit Secrets.")

            chunk_text, segments = transcribe_chunk_groq(
                chunk_path=chunk_path,
                api_key=groq_key,
                time_offset=cumulative_offset
            )

        if chunk_text:
            full_text_list.append(chunk_text)
            all_segments.extend(segments)

        cumulative_offset += get_audio_duration(chunk_path)

    return {
        "full_text": " ".join(full_text_list),
        "segments": all_segments
    }


# ── App Wrapper ──────────────────────────────────────────────────────────────
def transcribe_all(chunks: List[str], language: str = "english") -> Dict[str, Any]:
    """App wrapper called by app.py."""
    provider = "sarvam" if language.lower() in ["hinglish", "sarvam"] else "groq"
    translate_flag = True if language.lower() == "hinglish" else False

    return process_transcription(
        chunks=chunks,
        provider=provider,
        translate_to_english=translate_flag
    )
