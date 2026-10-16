import io
import os
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

st.set_page_config(page_title="Voice AI Assistant", page_icon="🎙️", layout="centered")
st.title("🎙️ Voice AI Assistant")
st.write("Upload a recording → transcribe it → ask the AI → listen to its spoken response.")

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("OPENAI_API_KEY is missing. Add it to your .env file and restart the app.")
    st.stop()

client = OpenAI(api_key=api_key)

uploaded = st.file_uploader(
    "Upload a voice recording",
    type=["wav", "mp3", "m4a", "mpeg", "mpga", "webm", "mp4"],
    help="Use a clear recording. WAV, MP3, and M4A are recommended."
)

if uploaded:
    st.audio(uploaded)

    if st.button("Run Voice AI", type="primary"):
        overall_start = time.perf_counter()
        audio_bytes = uploaded.getvalue()

        if len(audio_bytes) < 1000:
            st.warning("This audio file seems very small or empty. Please upload a longer recording.")
            st.stop()

        transcript = ""
        answer = ""
        speech_bytes = b""
        stt_seconds = llm_seconds = tts_seconds = 0.0

        try:
            # 1) Speech-to-text
            with st.spinner("Transcribing your audio..."):
                stage_start = time.perf_counter()
                audio_file = (uploaded.name, io.BytesIO(audio_bytes), uploaded.type or "application/octet-stream")
                transcription = client.audio.transcriptions.create(
                    model="gpt-4o-mini-transcribe",
                    file=audio_file,
                )
                stt_seconds = time.perf_counter() - stage_start
                transcript = (getattr(transcription, "text", "") or "").strip()

            st.subheader("📝 Transcription")
            if not transcript:
                st.warning("I couldn't detect clear speech in this recording. Try a clearer or louder clip.")
                st.stop()
            st.write(transcript)

            # Guard against uncertain/noisy transcription
            if len(transcript) < 3:
                st.warning("The transcript is very short, so the AI response may be unreliable. Try recording again.")

            # 2) LLM response
            with st.spinner("Generating an AI response..."):
                stage_start = time.perf_counter()
                response = client.responses.create(
                    model="gpt-4o-mini",
                    instructions=(
                        "You are a helpful voice assistant. Answer the user's spoken request clearly, "
                        "naturally, and concisely. If the transcription appears incomplete or unclear, "
                        "say what you understood and ask one brief clarification. Do not invent details."
                    ),
                    input=transcript,
                )
                llm_seconds = time.perf_counter() - stage_start
                answer = (response.output_text or "").strip()

            if not answer:
                st.error("The AI returned an empty response. Please try again.")
                st.stop()

            st.subheader("🤖 AI Response")
            st.write(answer)

            # 3) Text-to-speech
            with st.spinner("Creating spoken audio..."):
                stage_start = time.perf_counter()
                speech = client.audio.speech.create(
                    model="gpt-4o-mini-tts",
                    voice="alloy",
                    input=answer,
                    response_format="mp3",
                )
                speech_bytes = speech.read()
                tts_seconds = time.perf_counter() - stage_start

            st.subheader("🔊 Listen to the response")
            st.audio(speech_bytes, format="audio/mp3")
            st.download_button(
                "Download AI response (MP3)",
                data=speech_bytes,
                file_name="voice_ai_response.mp3",
                mime="audio/mpeg",
            )

            total_seconds = time.perf_counter() - overall_start
            st.subheader("⏱️ Latency report")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("STT", f"{stt_seconds:.2f}s")
            col2.metric("LLM", f"{llm_seconds:.2f}s")
            col3.metric("TTS", f"{tts_seconds:.2f}s")
            col4.metric("Total", f"{total_seconds:.2f}s")

            stages = {
                "Speech-to-text": stt_seconds,
                "LLM response": llm_seconds,
                "Text-to-speech": tts_seconds,
            }
            bottleneck = max(stages, key=stages.get)
            st.caption(f"Slowest stage in this run: {bottleneck} ({stages[bottleneck]:.2f}s). "
                      "Latency varies with audio length, network, model load, and API response time.")

        except Exception as exc:
            st.error("The voice pipeline could not finish.")
            st.caption(
                "Check your API key, API billing/credits, internet connection, and supported audio format. "
                "If the audio is unclear, try a clearer recording."
            )
            st.code(str(exc))
