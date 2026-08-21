# AI Video Assistant 🎥🤖

An AI-powered application built with **Streamlit**, **LangChain**, and **yt-dlp** that extracts, transcribes, and indexes YouTube videos or uploaded media files for interactive RAG (Retrieval-Augmented Generation) Q&A and summarization.

---

## 🌟 Key Features

* **YouTube & Local Media Support:** Process YouTube video URLs directly or upload local audio/video files (`.mp3`, `.wav`, `.mp4`, `.mkv`).
* **Cloud IP Bypass:** Configured with `yt-dlp` mobile extractors and cookie secrets management to prevent `403 Forbidden` errors on hosted environments like Streamlit Community Cloud.
* **Audio Chunking:** Automatically breaks long media files into 10-minute segments using `pydub` to fit downstream API context windows and limits.
* **Vector Indexing & RAG:** Embeds video transcriptions into **ChromaDB** using HuggingFace sentence transformers for fast semantic search.
* **Multi-LLM Integration:** Powered by **LangChain** with support for Groq, Mistral AI, and Google Gemini models.

---

## 🛠️ Project Structure

```text
.
├── app.py                   # Streamlit main UI & routing
├── requirements.txt         # Python dependencies
├── utils/
│   └── audio_processor.py   # yt-dlp downloader, chunker, & format handler
└── README.md
