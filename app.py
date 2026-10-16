
import time
import tempfile
from pathlib import Path

import streamlit as st
import ollama
import pyttsx3
from faster_whisper import WhisperModel

st.set_page_config(
    page_title="VoxAI | Voice Assistant",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------- PREMIUM DARK UI ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

.stApp {
    background:
        radial-gradient(ellipse at top left, #20204b 0%, transparent 42%),
        radial-gradient(ellipse at top right, #123b50 0%, transparent 35%),
        #080b14;
    color: #f4f6ff;
    font-family: 'Inter', sans-serif;
}

.block-container {
    max-width: 1120px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

#MainMenu, footer, header {visibility: hidden;}

.hero {
    background: linear-gradient(120deg, #171b36, #102b3a);
    border: 1px solid #343b60;
    border-radius: 24px;
    padding: 35px;
    margin-bottom: 28px;
    box-shadow: 0 15px 50px #00000030;
}

.eyebrow {
    color: #69e2ff;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 3px;
}

.hero h1 {
    color: white;
    font-size: clamp(34px, 5vw, 54px);
    font-weight: 800;
    margin: 12px 0;
}

.hero p {
    color: #b9c4df;
    font-size: 16px;
    line-height: 1.8;
}

.pill {
    display: inline-block;
    padding: 7px 13px;
    border-radius: 30px;
    background: #1b3b3b;
    color: #85f5cb;
    border: 1px solid #28645c;
    font-size: 12px;
    font-weight: 700;
}

.section-title {
    font-size: 21px;
    font-weight: 750;
    color: #f4f6ff;
    margin: 18px 0 8px 0;
}

div[data-testid="stFileUploader"] {
    background: #111827;
    border: 1px dashed #5867a0;
    border-radius: 18px;
    padding: 15px;
}

div[data-testid="stFileUploader"] section {
    background: transparent;
}

.stButton > button,
.stDownloadButton > button {
    border: 0;
    border-radius: 12px;
    min-height: 46px;
    font-weight: 750;
    background: linear-gradient(100deg, #7767ff, #24c9e8);
    color: #080b14;
    transition: 0.2s ease;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    box-shadow: 0 0 24px #7667ff65;
    border: 0;
    transform: translateY(-1px);
}

div[data-testid="stAudio"] {
    background: #141a2b;
    padding: 10px;
    border-radius: 14px;
}

div[data-testid="stMetric"] {
    background: linear-gradient(145deg, #171d30, #101523);
    border: 1px solid #2b3551;
    padding: 18px;
    border-radius: 17px;
}

div[data-testid="stMetricLabel"] {
    color: #aab7d5;
}

div[data-testid="stMetricValue"] {
    color: #71e5ff;
}

div[data-testid="stAlert"] {
    border-radius: 13px;
}

hr {
    border-color: #252d43;
}

.footer {
    text-align: center;
    color: #7e8ba8;
    font-size: 12px;
    padding-top: 30px;
}
</style>
""", unsafe_allow_html=True)

# ---------- HERO ----------
st.markdown("""
<div class="hero">
    <div class="eyebrow">NEXT-GEN • SPEECH INTELLIGENCE</div>
    <h1>🎙️ VoxAI Assistant</h1>
    <p>
        Your voice, understood by AI. Upload a recording, convert speech
        into text, generate an intelligent response, and listen to it.
    </p>
    <span class="pill">● LOCAL AI PIPELINE</span>
    &nbsp;&nbsp;
    <span class="pill">STT → LLM → TTS</span>
</div>
""", unsafe_allow_html=True)

left, right = st.columns([1, 1], gap="large")

@st.cache_resource
def load_whisper():
    return WhisperModel("base", device="cpu", compute_type="int8")

with left:
    st.markdown('<div class="section-title">01 / Audio Input</div>',
                unsafe_allow_html=True)
    st.write("Upload a clear voice recording to start.")

    uploaded = st.file_uploader(
        "Choose your audio file",
        type=["wav", "mp3", "m4a", "mpeg", "mpga", "webm", "mp4"],
    )

    if uploaded:
        st.audio(uploaded)
        st.success(f"Audio ready: {uploaded.name}")

    run_button = st.button(
        "✨  Run Voice AI Pipeline",
        type="primary",
        use_container_width=True,
        disabled=uploaded is None,
    )

with right:
    st.markdown('<div class="section-title">02 / How It Works</div>',
                unsafe_allow_html=True)
    st.markdown("""
    **🎧 Speech-to-Text**  
    Convert your recording into readable text.

    **🧠 Local AI Brain**  
    Generate a useful response with Ollama.

    **🔊 Text-to-Speech**  
    Listen to the AI answer and download it.

    **🔒 Local Processing**  
    Whisper, Ollama and pyttsx3 run locally.
    """)

if run_button and uploaded:
    total_start = time.perf_counter()
    stt_seconds = llm_seconds = tts_seconds = 0.0
    speech_bytes = None

    try:
        audio_bytes = uploaded.getvalue()

        if len(audio_bytes) < 1000:
            st.error("Audio file is empty or too small.")
            st.stop()

        st.divider()
        st.markdown("### ⚡ Processing your voice")

        # 1. Speech to text
        with st.spinner("🎧 Listening and transcribing..."):
            start = time.perf_counter()
            suffix = Path(uploaded.name).suffix or ".wav"

            with tempfile.NamedTemporaryFile(
                delete=False, suffix=suffix
            ) as audio_file:
                audio_file.write(audio_bytes)
                audio_path = audio_file.name

            try:
                model = load_whisper()
                segments, info = model.transcribe(
                    audio_path, beam_size=5
                )
                transcript = " ".join(
                    segment.text.strip() for segment in segments
                ).strip()
            finally:
                Path(audio_path).unlink(missing_ok=True)

            stt_seconds = time.perf_counter() - start

        if not transcript:
            st.warning("No clear speech detected. Try another recording.")
            st.stop()

        st.markdown("### 📝 Your Transcription")
        with st.container(border=True):
            st.write(transcript)

        # 2. Local LLM
        with st.spinner("🧠 Generating an intelligent response..."):
            start = time.perf_counter()
            response = ollama.chat(
                model="llama3.2:3b",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are VoxAI, a helpful voice assistant. "
                            "Answer naturally, clearly and concisely. "
                            "If a request is unclear, ask one short question."
                        ),
                    },
                    {"role": "user", "content": transcript},
                ],
            )
            answer = response["message"]["content"].strip()
            llm_seconds = time.perf_counter() - start

        if not answer:
            st.error("The AI returned an empty response.")
            st.stop()

        st.markdown("### 🤖 VoxAI Response")
        with st.container(border=True):
            st.write(answer)

        # 3. Text to speech
        with st.spinner("🔊 Creating your AI voice response..."):
            start = time.perf_counter()
            engine = pyttsx3.init()
            engine.setProperty("rate", 170)

            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".wav"
            ) as speech_file:
                speech_path = speech_file.name

            try:
                engine.save_to_file(answer, speech_path)
                engine.runAndWait()

                with open(speech_path, "rb") as file:
                    speech_bytes = file.read()
            finally:
                engine.stop()
                Path(speech_path).unlink(missing_ok=True)

            tts_seconds = time.perf_counter() - start

        if not speech_bytes:
            st.error("Audio generation returned an empty file.")
            st.stop()

        st.markdown("### 🔊 Listen to Your AI")
        st.audio(speech_bytes, format="audio/wav")

        st.download_button(
            "⬇️ Download AI Voice (WAV)",
            data=speech_bytes,
            file_name="voxai_response.wav",
            mime="audio/wav",
        )

        total_seconds = time.perf_counter() - total_start

        st.divider()
        st.markdown("### 📊 Performance Dashboard")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🎧 STT", f"{stt_seconds:.2f}s")
        c2.metric("🧠 LLM", f"{llm_seconds:.2f}s")
        c3.metric("🔊 TTS", f"{tts_seconds:.2f}s")
        c4.metric("⚡ Total", f"{total_seconds:.2f}s")

        st.success("Voice pipeline completed successfully!")

    except Exception as exc:
        st.error("The voice pipeline could not finish.")
        st.code(str(exc))
        st.info(
            "Check that Ollama is running and llama3.2:3b is installed. "
            "Also verify the audio file and available memory."
        )

st.markdown("""
<div class="footer">
    VOXAI • BUILT WITH PYTHON, WHISPER, OLLAMA & STREAMLIT
</div>
""", unsafe_allow_html=True)
