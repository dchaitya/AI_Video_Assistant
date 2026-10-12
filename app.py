import os
import sys
import time
import json
import logging
import tempfile

# Suppress noisy Streamlit file watcher warnings when inspecting heavy ML libraries (e.g. transformers, torchvision)
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
logging.getLogger("streamlit.watcher.local_sources_watcher").setLevel(logging.ERROR)

import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# If running on Streamlit Community Cloud, load secrets into environment variables
try:
    if "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except Exception:
    pass

# Add root directory to sys.path so core and utils can be imported reliably
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from utils.audio_processor import (
    process_input,
    download_yt_audio,
    convert_to_wav,
    chunk_audio,
    get_audio_path,
)
from core.transcriber import transcribe_all, get_groq_client
from core.summarize import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question, ask_question_with_sources
from core.vector_store import build_vector_store, get_retriever

# ─────────────────────────────────────────────────────────────────────────────
# Streamlit Page Config & Custom Styling
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Video & Meeting Assistant",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern styling
st.markdown(
    """
    <style>
    /* Metric Card Styling */
    .metric-card {
        background: rgba(128, 128, 128, 0.08);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    .metric-value {
        font-size: 24px;
        font-weight: 700;
        margin-top: 4px;
        color: #ff4b4b;
    }
    .metric-label {
        font-size: 13px;
        opacity: 0.8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Hero Banner */
    .hero-container {
        padding: 20px 24px;
        border-radius: 12px;
        background: linear-gradient(135deg, rgba(255, 75, 75, 0.1) 0%, rgba(30, 144, 255, 0.08) 100%);
        border: 1px solid rgba(255, 75, 75, 0.2);
        margin-bottom: 24px;
    }
    .hero-title {
        font-size: 28px;
        font-weight: 800;
        margin-bottom: 6px;
    }
    .hero-subtitle {
        font-size: 15px;
        opacity: 0.85;
        line-height: 1.5;
    }

    /* Section Cards */
    .content-box {
        background: rgba(128, 128, 128, 0.05);
        border: 1px solid rgba(128, 128, 128, 0.18);
        border-radius: 10px;
        padding: 20px;
        margin-top: 10px;
    }

    /* Badges */
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 16px;
        font-size: 12px;
        font-weight: 600;
        margin-right: 6px;
        background: rgba(255, 75, 75, 0.15);
        color: #ff4b4b;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Session State Initialization
# ─────────────────────────────────────────────────────────────────────────────
if "processed_data" not in st.session_state:
    st.session_state.processed_data = None
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "youtube_url_input" not in st.session_state:
    st.session_state.youtube_url_input = ""

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar: Settings & Configuration
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎬 AI Video Assistant")
    st.caption("Meeting & Video Intelligence via RAG")
    st.divider()

    # Model & Parameter Settings
    st.markdown("#### ⚙️ Pipeline Settings")
    model_choice = st.selectbox(
        "LLM Model",
        options=["qwen/qwen3.8-27b", "openai/gpt-oss-120b", "openai/gpt-oss-20b"],
        index=0,
        help="Fast and capable Groq-hosted LLMs",
    )

    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.05,
        help="Lower values yield more factual, deterministic outputs.",
    )

    chunk_minutes = st.slider(
        "Audio Chunk Size (minutes)",
        min_value=2,
        max_value=10,
        value=5,
        step=1,
        help="Size of audio segments split for Whisper transcription.",
    )

    rag_k = st.slider(
        "RAG Top-K Context Chunks",
        min_value=2,
        max_value=8,
        value=4,
        step=1,
        help="Number of relevant transcript passages retrieved per question.",
    )

    st.divider()

    # Clear / Reset Button
    if st.button("🔄 Reset / Clear Session", use_container_width=True):
        st.session_state.processed_data = None
        st.session_state.rag_chain = None
        st.session_state.retriever = None
        st.session_state.chat_history = []
        st.session_state.youtube_url_input = ""
        st.rerun()

    st.divider()
    st.caption("Built with Groq Whisper • LangChain • ChromaDB • Streamlit")


# ─────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────────────────────
def format_seconds(seconds: int) -> str:
    mins, secs = divmod(int(seconds), 60)
    hours, mins = divmod(mins, 60)
    if hours > 0:
        return f"{hours}h {mins}m {secs}s"
    return f"{mins}m {secs}s"


def create_markdown_report(data: dict) -> str:
    """Generates an exportable comprehensive Markdown report."""
    md = f"""# {data.get('title', 'Meeting Intelligence Report')}

**Source:** {data.get('source_name', 'Uploaded File')}  
**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}  
**Total Words:** {data.get('word_count', 0)}  
**Audio Chunks:** {data.get('chunks_count', 0)}  

---

## 📋 Executive Summary
{data.get('summary', 'No summary generated.')}

---

## ✅ Action Items
{data.get('action_items', 'No action items found.')}

---

## 🔑 Key Decisions
{data.get('key_decisions', 'No key decisions found.')}

---

## ❓ Open Questions & Follow-ups
{data.get('open_questions', 'No open questions found.')}

---

## 📝 Full Transcript
{data.get('transcript', '')}
"""
    return md


# ─────────────────────────────────────────────────────────────────────────────
# Main View: Input & Processing or Results Dashboard
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.processed_data is None:
    # ── HERO BANNER ──
    st.markdown(
        """
        <div class="hero-container">
            <div class="hero-title">🎬 AI Video & Meeting Intelligence Assistant</div>
            <div class="hero-subtitle">
                Transform long YouTube videos and audio/video recordings into instant executive summaries, 
                action items, key decisions, and an interactive conversational knowledge base powered by RAG.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── INPUT TABS ──
    input_tab1, input_tab2 = st.tabs(["🌐 YouTube Video", "📁 Upload Audio / Video File"])

    source_path = None
    source_name = None
    source_type = None

    with input_tab1:
        st.markdown("#### Enter YouTube Video URL")
        col_url, col_sample = st.columns([4, 1])

        with col_url:
            yt_url = st.text_input(
                "YouTube URL",
                placeholder="https://www.youtube.com/watch?v=...",
                value=st.session_state.youtube_url_input,
                label_visibility="collapsed",
            )
        with col_sample:
            if st.button("Load Sample Video", use_container_width=True):
                st.session_state.youtube_url_input = "https://www.youtube.com/watch?v=M64HUIJFTZM"
                st.rerun()

        if yt_url.strip():
            source_path = yt_url.strip()
            source_name = yt_url.strip()
            source_type = "youtube"
            try:
                st.video(yt_url.strip())
            except Exception:
                pass

    with input_tab2:
        st.markdown("#### Upload Meeting Recording or Podcast")
        uploaded_file = st.file_uploader(
            "Choose an audio or video file",
            type=["mp4", "mkv", "mp3", "wav", "m4a", "webm", "flac"],
            help="Supports popular video and audio formats up to 200MB",
        )

        if uploaded_file is not None:
            # Save uploaded file to downloads folder
            downloads_dir = "downloads"
            os.makedirs(downloads_dir, exist_ok=True)
            temp_path = os.path.join(downloads_dir, f"upload_{uploaded_file.name}")
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            source_path = temp_path
            source_name = uploaded_file.name
            source_type = "upload"

            st.success(f"File uploaded: **{uploaded_file.name}** ({uploaded_file.size / (1024*1024):.2f} MB)")
            if uploaded_file.type.startswith("audio"):
                st.audio(temp_path)
            elif uploaded_file.type.startswith("video"):
                st.video(temp_path)

    st.write("")
    process_btn = st.button("⚡ Process & Analyze Meeting", type="primary", use_container_width=True)

    if process_btn:
        if not os.getenv("GROQ_API_KEY"):
            st.error("GROQ_API_KEY is missing. Please set it in your .env file.")
        elif not source_path:
            st.warning("Please provide a YouTube URL or upload a file first.")
        else:
            try:
                with st.status("🚀 Processing Media & Building AI Knowledge Base...", expanded=True) as status:
                    # Step 1 & 2: Audio downloading / conversion & chunking
                    st.write("📥 Preparing and converting audio stream...")
                    chunks = process_input(source_path, chunk_minutes=chunk_minutes)
                    st.write(f"✂️ Created **{len(chunks)}** audio chunks for Whisper processing.")

                    # Step 3: Transcription
                    st.write("🎙️ Transcribing audio with Groq Whisper...")
                    progress_bar = st.progress(0.0)

                    def update_progress(current, total):
                        progress_bar.progress(current / total, text=f"Transcribing chunk {current} of {total}...")

                    transcript = transcribe_all(chunks, progress_callback=update_progress)
                    progress_bar.progress(1.0, text="Transcription completed!")
                    st.write(f"📝 Transcription complete: **{len(transcript.split())} words** extracted.")

                    # Step 4: Title Generation
                    st.write("🏷️ Generating meeting title...")
                    title = generate_title(transcript, model=model_choice)
                    st.write(f"📌 **{title}**")

                    # Step 5: Summarization
                    st.write("📋 Generating executive summary...")
                    summary = summarize(transcript, model=model_choice, temperature=temperature)

                    # Step 6: Information Extraction
                    st.write("🎯 Extracting action items, key decisions, and open questions...")
                    action_items = extract_action_items(transcript, model=model_choice, temperature=temperature)
                    key_decisions = extract_key_decisions(transcript, model=model_choice, temperature=temperature)
                    open_questions = extract_questions(transcript, model=model_choice, temperature=temperature)

                    # Step 7: Vector Store & RAG Chain
                    st.write("🧠 Indexing transcript in Chroma Vector Store for RAG chat...")
                    session_collection = f"col_{int(time.time())}"
                    vector_store = build_vector_store(transcript, collection_name=session_collection, clear_existing=False)
                    retriever = get_retriever(vector_store, k=rag_k)
                    rag_chain = build_rag_chain(
                        retriever=retriever,
                        model=model_choice,
                        temperature=temperature,
                    )

                    status.update(label="✅ Processing complete! Loading dashboard...", state="complete", expanded=False)

                # Save all results to session state
                st.session_state.collection_name = session_collection
                st.session_state.processed_data = {
                    "title": title,
                    "transcript": transcript,
                    "summary": summary,
                    "action_items": action_items,
                    "key_decisions": key_decisions,
                    "open_questions": open_questions,
                    "source_name": source_name,
                    "source_type": source_type,
                    "chunks_count": len(chunks),
                    "word_count": len(transcript.split()),
                    "char_count": len(transcript),
                }
                st.session_state.rag_chain = rag_chain
                st.session_state.retriever = retriever
                st.session_state.chat_history = [
                    {
                        "role": "assistant",
                        "content": f"Hello! I have fully processed **{title}**. Ask me anything about this meeting or video!",
                        "sources": [],
                    }
                ]
                st.rerun()

            except Exception as e:
                st.error(f"❌ An error occurred during processing: {str(e)}")
                st.info("Tip: Ensure you have a valid GROQ_API_KEY in your .env file, internet connection, and ffmpeg installed.")

else:
    # ─────────────────────────────────────────────────────────────────────────
    # RESULTS DASHBOARD
    # ─────────────────────────────────────────────────────────────────────────
    data = st.session_state.processed_data

    # Header Card
    col_head, col_btn = st.columns([4, 1])
    with col_head:
        st.markdown(f"## 📌 {data.get('title', 'Meeting Analysis')}")
        st.caption(f"Source: `{data.get('source_name', 'Unknown')}`")
    with col_btn:
        if st.button("➕ Analyze Another", use_container_width=True):
            st.session_state.processed_data = None
            st.session_state.rag_chain = None
            st.session_state.retriever = None
            st.session_state.chat_history = []
            st.session_state.youtube_url_input = ""
            st.rerun()

    # Metric Badges Row
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Words Extracted</div>
                <div class="metric-value">{data.get('word_count', 0):,}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_m2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Audio Chunks</div>
                <div class="metric-value">{data.get('chunks_count', 0)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_m3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Total Characters</div>
                <div class="metric-value">{data.get('char_count', 0):,}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_m4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">RAG Chat Status</div>
                <div class="metric-value" style="color: #00c853;">READY</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")

    # Dashboard Tabs
    tab_sum, tab_action, tab_dec, tab_q, tab_chat, tab_trans, tab_export = st.tabs(
        [
            "📋 Summary",
            "✅ Action Items",
            "🔑 Key Decisions",
            "❓ Open Questions",
            "💬 Chat (RAG)",
            "📝 Transcript",
            "📥 Export Report",
        ]
    )

    # 1. Summary Tab
    with tab_sum:
        st.markdown("### 📋 Executive Summary")
        st.markdown(data.get("summary", "No summary available."))

    # 2. Action Items Tab
    with tab_action:
        st.markdown("### ✅ Action Items & Tasks")
        st.markdown(data.get("action_items", "No action items extracted."))

    # 3. Key Decisions Tab
    with tab_dec:
        st.markdown("### 🔑 Key Decisions")
        st.markdown(data.get("key_decisions", "No key decisions extracted."))

    # 4. Open Questions Tab
    with tab_q:
        st.markdown("### ❓ Open Questions & Follow-ups")
        st.markdown(data.get("open_questions", "No open questions extracted."))

    # 5. Interactive RAG Chat Tab
    with tab_chat:
        st.markdown("### 💬 Chat with Your Meeting")
        st.caption("Ask specific questions. Answers are grounded exclusively in the transcript via Chroma Vector Store.")

        # Quick Prompt Suggestions
        st.markdown("**Suggested Prompts:**")
        sug_cols = st.columns(4)
        sample_prompts = [
            "What are the main takeaways?",
            "Who has assigned tasks and deadlines?",
            "What key decisions were reached?",
            "What were the controversial points?",
        ]

        selected_prompt = None
        for idx, prompt_text in enumerate(sample_prompts):
            with sug_cols[idx]:
                if st.button(prompt_text, key=f"sug_{idx}", use_container_width=True):
                    selected_prompt = prompt_text

        st.divider()

        # Render Chat History
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message.get("sources"):
                    with st.expander("🔍 View Retrieved Context Snippets"):
                        for i, doc in enumerate(message["sources"]):
                            st.markdown(f"**Passage {i+1}:**")
                            st.info(doc.page_content)

        # Chat Input
        user_input = st.chat_input("Ask a question about this meeting...")
        prompt_to_run = selected_prompt or user_input

        if prompt_to_run:
            # Display user message
            st.session_state.chat_history.append({"role": "user", "content": prompt_to_run, "sources": []})
            with st.chat_message("user"):
                st.markdown(prompt_to_run)

            # Generate Assistant Answer via RAG
            with st.chat_message("assistant"):
                with st.spinner("Searching transcript context and thinking..."):
                    try:
                        retriever = st.session_state.get("retriever")
                        rag_chain = st.session_state.get("rag_chain")
                        col_name = st.session_state.get("collection_name")

                        # Self-healing: if session objects are missing, reconstruct cleanly
                        if retriever is None or rag_chain is None:
                            transcript_text = data.get("transcript", "")
                            if col_name:
                                try:
                                    vs = load_vector_store(col_name)
                                except Exception:
                                    vs = build_vector_store(transcript_text, collection_name=col_name, clear_existing=False)
                            else:
                                vs = build_vector_store(transcript_text, clear_existing=False)
                            retriever = get_retriever(vs, k=rag_k)
                            rag_chain = build_rag_chain(retriever=retriever, model=model_choice, temperature=temperature)
                            st.session_state.retriever = retriever
                            st.session_state.rag_chain = rag_chain

                        answer, docs = ask_question_with_sources(
                            rag_chain,
                            retriever,
                            prompt_to_run,
                            collection_name=col_name,
                        )
                        st.markdown(answer)

                        if docs:
                            with st.expander("🔍 View Retrieved Context Snippets"):
                                for i, doc in enumerate(docs):
                                    st.markdown(f"**Passage {i+1}:**")
                                    st.info(doc.page_content)

                        # Save to chat history
                        st.session_state.chat_history.append(
                            {"role": "assistant", "content": answer, "sources": docs}
                        )

                    except Exception as e:
                        err_msg = f"Failed to retrieve answer: {str(e)}"
                        st.error(err_msg)
                        st.session_state.chat_history.append(
                            {"role": "assistant", "content": err_msg, "sources": []}
                        )

        # Clear Chat button
        if len(st.session_state.chat_history) > 1:
            if st.button("🧹 Clear Chat History"):
                st.session_state.chat_history = [st.session_state.chat_history[0]]
                st.rerun()

    # 6. Full Transcript Tab
    with tab_trans:
        st.markdown("### 📝 Full Meeting Transcript")
        search_query = st.text_input("🔍 Search in transcript:", placeholder="Type a keyword to highlight...")

        transcript_text = data.get("transcript", "")
        if search_query.strip():
            count = transcript_text.lower().count(search_query.strip().lower())
            st.caption(f"Found **{count}** occurrences of `{search_query}`")

        st.text_area(
            "Raw Transcript",
            value=transcript_text,
            height=400,
            help="Complete verbatim transcription generated by Whisper",
        )

        st.download_button(
            label="📥 Download Full Transcript (.txt)",
            data=transcript_text,
            file_name=f"{data.get('title', 'transcript')}.txt",
            mime="text/plain",
            use_container_width=True,
        )

    # 7. Export Report Tab
    with tab_export:
        st.markdown("### 📥 Export Intelligence Report")
        st.caption("Download the complete synthesis (Summary, Action Items, Decisions, Questions, and Transcript).")

        report_md = create_markdown_report(data)
        col_exp1, col_exp2 = st.columns(2)

        with col_exp1:
            st.download_button(
                label="📄 Download Report as Markdown (.md)",
                data=report_md,
                file_name=f"{data.get('title', 'meeting_report')}.md",
                mime="text/markdown",
                use_container_width=True,
            )

        with col_exp2:
            st.download_button(
                label="📑 Download Report as Plain Text (.txt)",
                data=report_md,
                file_name=f"{data.get('title', 'meeting_report')}.txt",
                mime="text/plain",
                use_container_width=True,
            )

        st.write("")
        st.markdown("#### Preview Export Markdown:")
        with st.expander("👀 View Full Report Content", expanded=False):
            st.code(report_md, language="markdown")

