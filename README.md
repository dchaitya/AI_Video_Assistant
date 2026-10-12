# 🎬 AI Video & Meeting Assistant (RAG Pipeline)

An end-to-end AI assistant that transcribes, summarizes, extracts action items and key decisions, and lets you chat with any YouTube video or recorded audio/video file using Retrieval-Augmented Generation (RAG).

---

## 🌟 Key Features

- **Dual Ingestion Options**:
  - 🌐 **YouTube URLs**: Automatic audio extraction and downloading.
  - 📁 **Local Audio & Video Files**: Drag-and-drop upload for `.mp4`, `.mp3`, `.wav`, `.m4a`, `.webm`, `.mkv`, and `.flac`.
- **Speech-to-Text**:
  - Accelerated Whisper transcription via Groq Cloud (`whisper-large-v3`).
  - Automatic audio chunking into configurable segments to handle long meetings.
- **Meeting Intelligence Extraction**:
  - 🏷️ **AI Title Generation**: Short, descriptive titles.
  - 📋 **Executive Summary**: Map-Reduce structured summary in bullet points.
  - ✅ **Action Items**: Task description, responsible owner, and deadline.
  - 🔑 **Key Decisions**: Decisions reached during the meeting.
  - ❓ **Open Questions**: Unresolved topics and follow-up items.
- **Conversational RAG Chat**:
  - Indexing with Chroma Vector Store and HuggingFace MiniLM embeddings.
  - Ask questions strictly grounded in the meeting transcript.
  - View retrieved context snippets for complete transparency and verification.
- **Export & Download**:
  - Export full intelligence reports as Markdown (`.md`) or Plain Text (`.txt`).
  - Download full verbatim transcript (`.txt`).

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have **Python >= 3.10** and **FFmpeg** installed on your system.

### 2. Install Dependencies
```bash
# If using a virtual environment (recommended):
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

### 3. API Key Setup
Create or update your `.env` file in the project root:
```env
GROQ_API_KEY="your_groq_api_key_here"
```
*(You can get a free API key at [console.groq.com](https://console.groq.com/keys)).*

---

## 💻 Running the Application

### 🖥️ Streamlit Web UI (Recommended)
Launch the interactive web interface:
```bash
.\.venv\Scripts\streamlit.exe run app.py
```
Or with standard Python environment:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### ⌨️ Command-Line Interface (CLI)
You can also run the pipeline directly via terminal:
```bash
.\.venv\Scripts\python.exe main.py
```

---

## ☁️ Deploying to Streamlit Community Cloud

1. **Push your code to GitHub**:
   ```bash
   git init
   git add .
   git commit -m "Deploy AI Video Assistant"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<your-repo-name>.git
   git push -u origin main
   ```
   *(The included `.gitignore` will ensure your `.env`, `.venv`, and `downloads` remain private.)*

2. **Deploy on Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io/) and click **"New app"**.
   - Select your repository, set Branch to `main`, and Main file path to `app.py`.
   - Click **"Advanced settings"** -> **"Secrets"**, and enter:
     ```toml
     GROQ_API_KEY = "your_actual_groq_api_key_here"
     ```
   - Click **Deploy**!
   *(System dependencies like `ffmpeg` are automatically installed from `packages.txt`.)*

---

## 📁 Project Structure

```text
AI_video_assistant_RAG_project/
├── app.py                   # Streamlit Web UI
├── main.py                  # CLI pipeline entry point
├── core/
│   ├── extractor.py         # Extracts action items, decisions & questions
│   ├── rag_engine.py        # LangChain RAG chain & QA engine
│   ├── summarize.py         # Map-Reduce meeting summarization
│   ├── transcriber.py       # Groq Whisper STT transcriber
│   └── vector_store.py      # Chroma vector database & embeddings
├── utils/
│   └── audio_processor.py   # YouTube download, audio conversion & chunking
├── downloads/               # Directory for downloaded / converted audio files
├── vector_db/               # Persistent Chroma vector store directory
├── requirements.txt         # Project dependencies
└── .env                     # API keys and environment variables
```

