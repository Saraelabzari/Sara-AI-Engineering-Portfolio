import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st
import whisper
from transformers import pipeline

APP_TITLE = "Multilingual Speech-to-Text & Summarizer"
SUMMARY_MODEL = "csebuetnlp/mT5_multilingual_XLSum"
SUPPORTED_EXTENSIONS = ["wav", "mp3", "m4a", "mp4", "mpeg", "mpga", "webm", "ogg", "flac"]
LANGUAGE_CODES = {
    "English": "en", "Arabic": "ar", "Malay": "ms", "French": "fr",
    "Spanish": "es", "German": "de", "Hindi": "hi", "Chinese": "zh",
    "Japanese": "ja", "Portuguese": "pt", "Russian": "ru", "Turkish": "tr", "Urdu": "ur",
}

st.set_page_config(page_title=APP_TITLE, page_icon="🎙️", layout="wide")
st.title("🎙️ Multilingual Speech-to-Text & Summarizer")
st.write("Transcribe multilingual audio, generate a concise summary, and export results.")

with st.sidebar:
    st.header("Settings")
    model_name = st.selectbox("Whisper model", ["tiny", "base", "small", "medium"], index=1)
    language = st.selectbox("Speech language", ["Auto-detect", *LANGUAGE_CODES.keys()])
    task = st.radio("Audio task", ["Transcribe in original language", "Translate speech to English"])
    enable_summary = st.checkbox("Generate summary", value=True)
    timestamps = st.checkbox("Include segment timestamps", value=True)
    st.caption("Models run locally after download. First run requires internet access.")
    st.caption("Only upload recordings you are authorized to process.")

@st.cache_resource(show_spinner="Loading Whisper model...")
def load_whisper(name):
    return whisper.load_model(name)

@st.cache_resource(show_spinner="Loading multilingual summarizer...")
def load_summarizer():
    return pipeline("text2text-generation", model=SUMMARY_MODEL, tokenizer=SUMMARY_MODEL)

from src.audio_utils import make_transcript, timestamp
from src.summarization import summarize_text as summarize


def heuristic_action_items(text):
    cues = ("action:", "to do:", "todo:", "follow up", "follow-up", "need to", "must ", "should ")
    items = []
    for sentence in text.replace("\n", " ").split(". "):
        sentence = sentence.strip(" •-\t")
        if sentence and any(cue in sentence.lower() for cue in cues):
            items.append(sentence.rstrip(".") + ".")
    return items[:8]

uploaded = st.file_uploader("Upload an audio file", type=SUPPORTED_EXTENSIONS)
if uploaded:
    st.audio(uploaded)
    st.caption(f"{uploaded.name} · {uploaded.size / (1024 * 1024):.1f} MB")
    if uploaded.size > 200 * 1024 * 1024:
        st.error("This demo limits files to 200 MB.")
        st.stop()
    if st.button("Transcribe audio", type="primary", use_container_width=True):
        temp_path = None
        try:
            suffix = Path(uploaded.name).suffix or ".audio"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
                temp.write(uploaded.getbuffer())
                temp_path = temp.name
            options = {"fp16": False}
            if language != "Auto-detect":
                options["language"] = LANGUAGE_CODES[language]
            if task == "Translate speech to English":
                options["task"] = "translate"
            with st.spinner("Transcribing audio..."):
                result = load_whisper(model_name).transcribe(temp_path, **options)
            transcript = make_transcript(result, timestamps)
            if not transcript:
                st.warning("No speech detected. Try clearer audio or another model.")
                st.stop()
            st.session_state["transcript"] = transcript
            st.session_state["detected_language"] = result.get("language", "unknown")
            st.session_state["source_file"] = uploaded.name
            st.session_state["summary"] = ""
            st.session_state["actions"] = []
        except Exception as exc:
            st.error("Transcription failed. Check FFmpeg, dependencies, and audio format.")
            with st.expander("Technical details"):
                st.code(str(exc))
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

if st.session_state.get("transcript"):
    transcript = st.session_state["transcript"]
    detected = st.session_state.get("detected_language", "unknown")
    st.subheader("Transcription results")
    a, b, c = st.columns(3)
    a.metric("Detected language", detected.upper())
    b.metric("Words", f"{len(transcript.split()):,}")
    c.metric("Characters", f"{len(transcript):,}")
    transcript_tab, summary_tab, export_tab = st.tabs(["Transcript", "Summary", "Export"])
    with transcript_tab:
        st.text_area("Transcript", transcript, height=320)
        st.download_button("Download transcript (.txt)", transcript, "transcript.txt", "text/plain")
    with summary_tab:
        if not enable_summary:
            st.info("Enable summary generation in the sidebar.")
        else:
            max_tokens = st.slider("Summary length (maximum new tokens)", 60, 220, 120, 20)
            if st.button("Generate summary", type="primary"):
                try:
                    with st.spinner("Generating summary..."):
                        summary = summarize(transcript, load_summarizer(), max_tokens)
                    st.session_state["summary"] = summary
                    st.session_state["actions"] = heuristic_action_items(summary)
                except Exception as exc:
                    st.error("Summary generation failed. Check internet access, model availability, and RAM.")
                    with st.expander("Technical details"):
                        st.code(str(exc))
            if st.session_state.get("summary"):
                st.markdown("#### AI-generated summary")
                st.write(st.session_state["summary"])
                st.caption("Review the summary for omissions or mistakes before relying on it.")
                st.markdown("#### Potential action items")
                actions = st.session_state.get("actions", [])
                if actions:
                    for item in actions:
                        st.markdown(f"- {item}")
                else:
                    st.info("No action items found by the simple keyword heuristic.")
    with export_tab:
        summary = st.session_state.get("summary", "")
        payload = {"project": APP_TITLE, "source_file": st.session_state.get("source_file", ""),
                   "created_at_utc": datetime.now(timezone.utc).isoformat(),
                   "detected_language_code": detected, "transcript": transcript, "summary": summary,
                   "potential_action_items": st.session_state.get("actions", [])}
        st.download_button("Download results (.json)", json.dumps(payload, ensure_ascii=False, indent=2),
                           "speech_summary_results.json", "application/json")
        combined = f"Language: {detected}\n\nSUMMARY\n{summary}\n\nTRANSCRIPT\n{transcript}\n"
        st.download_button("Download transcript and summary (.txt)", combined,
                           "speech_summary_results.txt", "text/plain")

st.divider()
st.caption("Portfolio prototype by Sara Elabzari · Local inference after model downloads · AI outputs may contain errors.")
