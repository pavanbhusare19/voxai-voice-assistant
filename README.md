# Voice AI Assistant

A voice-in / voice-out assistant built with Streamlit and OpenAI APIs.

## Pipeline

1. Upload a recorded audio clip.
2. Speech-to-text with `gpt-4o-mini-transcribe`.
3. Contextual response with `gpt-4o-mini`.
4. Text-to-speech with `gpt-4o-mini-tts`.
5. Play or download the MP3 response.
6. Measure STT, LLM, TTS, and total end-to-end latency.

## Run locally (Windows PowerShell)

```powershell
cd "$HOME\Downloads\voice-ai"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create `.env` and add your own API key:

```dotenv
OPENAI_API_KEY=your_openai_api_key_here
```

Start the app:

```powershell
streamlit run app.py
```

Open the local URL Streamlit prints (usually http://localhost:8501).

## Test plan

- Record a clear 5–15 second request and upload it.
- Try a noisy/mumbled recording; the app should warn or ask for clarification rather than silently echoing.
- Test a very short/empty file.
- Record the latency report for the demo and note the slowest stage.

## Demo recording

Record a short screen video showing: upload audio → transcription → AI response → playable audio → latency report.

Add your public demo link here before submitting:
`Demo: PASTE_YOUR_DEMO_LINK_HERE`

## GitHub

Do not commit `.env`, API keys, audio recordings, or `.venv`.
