# AI Video Assistant 🎥🤖

An AI-powered application built with **Streamlit**, **LangChain**, and **yt-dlp** that extracts, transcribes, and indexes YouTube videos or uploaded media files for interactive RAG (Retrieval-Augmented Generation) Q&A and summarization.

---


## 📌 Overview

**AI Video Assistant** is an end-to-end intelligent media analysis system. By leveraging advanced Retrieval-Augmented Generation (RAG) and high-performance LLMs, it allows users to convert lengthy video or audio content into structured summaries, actionable insights, and conversational knowledge bases.

The system processes media links or local files, handles chunking and speech-to-text processing, stores vector embeddings in **ChromaDB**, and provides real-time contextual Q&A.

---

## ✨ Features

- 🎥 **YouTube & Local Media Support:** Process YouTube URLs directly or upload `.mp3`, `.wav`, `.mp4`, or `.mkv` files.
- ⚡ **Cloud IP Bypass:** Integrated `yt-dlp` mobile extractor configurations to prevent `403 Forbidden` blocks on hosted platforms.
- ✂️ **Automatic Audio Chunking:** Splits long audio files into optimal 10-minute segments using `pydub` for API payload compliance.
- 🔍 **Vector Search & RAG:** Embeds transcripts into **ChromaDB** using HuggingFace sentence transformers for semantic query retrieval.
- 🤖 **Multi-LLM Support:** Dynamic orchestration with Groq, Mistral AI, and Google Gemini via **LangChain**.
- 📝 **Automated Summarization:** Generate instant key takeaways, timestamps, and structured outlines.
- 🌙 **Responsive Streamlit Interface:** Interactive and user-friendly web interface for seamless workflow.

---

## 🧠 System Architecture

| Property | Details |
|----------|---------|
| **UI Framework** | Streamlit |
| **Orchestration** | LangChain |
| **Vector Database** | ChromaDB |
| **Embeddings** | HuggingFace (`sentence-transformers`) |
| **Media Extraction** | `yt-dlp` & `pydub` (FFmpeg) |
| **Supported Input** | YouTube URLs, Local Audio/Video Files |

---

## 🛠️ Tech Stack

- Python 3.11
- Streamlit
- LangChain
- ChromaDB
- yt-dlp
- PyDub
- Groq / Mistral AI / Google Gemini APIs

---

## 📂 Project Structure

```text
ai-video-assistant/
│
├── core/                    # Core AI pipelines & transcription modules
├── utils/                   # Audio extraction, chunking, & yt-dlp helpers
├── app.py                   # Main Streamlit web application
├── main.py                  # Entry point / CLI script execution
├── test.py                  # Testing & assertion utilities
├── packages.txt             # System-level dependencies (e.g., ffmpeg)
├── requirements.txt         # Python package dependencies
├── runtime.txt              # Environment runtime specification (Python 3.11)
├── .gitignore               # Ignored files & environment rules
└── README.md                # Project documentation
