"""
Calling Agent Workflow - Streamlit app (end-to-end example)

Features included (all in one file):
- Streamlit UI to:
  - Simulate a call by uploading an audio file (WAV/MP3)
  - Start a Twilio outbound call (example code; requires Twilio account and public webhook)
  - View transcripts, sentiment, keywords, screening decisions and agent feedback
- ASR transcription using OpenAI's Whisper (python package) if available
  - Fallback to an offline silence if whisper not installed, or to a naive placeholder
- Sentiment analysis using Hugging Face transformers pipeline
- Keyword extraction using YAKE (if available) or TF-IDF fallback
- Simple decision engine that flags tone issues, candidate willingness, and recommends closing lines
- Example Flask webhook for Twilio to POST a recording URL (separate process)

NOTES:
- This is a blueprint. For production you should secure credentials, host webhooks on HTTPS (ngrok or real domain), and follow privacy laws.
- Replace placeholders for TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, OPENAI API or install local whisper.

Run instructions (quick):
1) Create a virtualenv and `pip install -r requirements.txt` with packages: streamlit, pydub, transformers, torch, yake, twilio, flask, nltk, whisper (optional)
2) Run Streamlit: `streamlit run calling_agent_streamlit.py`
3) (Optional) Run webhook server for Twilio: `python calling_agent_streamlit.py --run-webhook`

"""

import os
import io
import argparse
import tempfile
import threading
import json
from typing import List, Dict, Any

import streamlit as st
from pydub import AudioSegment

# Optional imports - we'll handle missing packages gracefully
try:
    import whisper
    HAVE_WHISPER = True
except Exception:
    HAVE_WHISPER = False

try:
    from transformers import pipeline
    HAVE_TRANSFORMERS = True
except Exception:
    HAVE_TRANSFORMERS = False

try:
    import yake
    HAVE_YAKE = True
except Exception:
    HAVE_YAKE = False

# Twilio & Flask for webhook demonstration
try:
    from twilio.rest import Client
    from flask import Flask, request
    HAVE_TWILIO = True
except Exception:
    HAVE_TWILIO = False

# ---------- Utility functions ----------

def save_uploaded_audio(uploaded_file) -> str:
    """Save a Streamlit UploadedFile to a temporary WAV file and return path."""
    suffix = os.path.splitext(uploaded_file.name)[1].lower()
    audio_bytes = uploaded_file.read()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(audio_bytes)
    tmp.flush()
    tmp.close()
    # convert to wav for processing
    out_wav = tmp.name + ".wav"
    try:
        sound = AudioSegment.from_file(tmp.name)
        sound.export(out_wav, format="wav")
    except Exception:
        # If pydub fails, assume the file already is wav
        out_wav = tmp.name
    return out_wav


def transcribe_with_whisper(audio_path: str) -> Dict[str, Any]:
    """Transcribe using whisper (if installed). Returns dict with 'text'."""
    if not HAVE_WHISPER:
        return {"text": ""}
    model = whisper.load_model("small")
    result = model.transcribe(audio_path)
    return result


def simple_transcribe_placeholder(audio_path: str) -> Dict[str, Any]:
    """Fallback placeholder transcription (not accurate)."""
    # In a non-demo environment replace with actual transcription service (OpenAI/whisper/huggingface)
    return {"text": "[Transcription placeholder] Could not run whisper. Please install 'whisper' package or enable OpenAI transcription."}


def load_sentiment_pipeline():
    if not HAVE_TRANSFORMERS:
        return None
    return pipeline("sentiment-analysis")


def get_sentiment(text: str, pipe) -> Dict[str, Any]:
    if pipe is None:
        return {"label": "NEUTRAL", "score": 0.5}
    out = pipe(text[:512])  # limit length
    if isinstance(out, list) and len(out) > 0:
        return out[0]
    return {"label": "NEUTRAL", "score": 0.5}


def extract_keywords_yake(text: str, max_keywords: int = 8):
    if HAVE_YAKE:
        kw_extractor = yake.KeywordExtractor(lan="en", n=1, top=max_keywords)
        keywords = kw_extractor.extract_keywords(text)
        return [k for k, _ in keywords]
    # Fallback: naive top nouns/adjectives using simple frequency
    words = [w.lower().strip(".,!?") for w in text.split() if len(w) > 3]
    freq = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    sorted_words = sorted(freq.items(), key=lambda x: -x[1])[:max_keywords]
    return [w for w, _ in sorted_words]


# ---------- Decision engine (screening logic) ----------

SCREENING_KEYWORDS = {
    'willing_to_join': ['join', 'start', 'notice period', 'available', 'immediately', 'relocate'],
    'skills': ['python', 'sql', 'aws', 'spark', 'machine learning', 'ml', 'data science', 'streamlit', 'nlp'],
    'experience': ['years', 'experience', 'worked', 'projects', 'lead']
}


def decision_engine(transcript_text: str, sentiment_result: Dict[str, Any], keywords: List[str]) -> Dict[str, Any]:
    """Simple heuristic-based decision engine."""
    text = transcript_text.lower()
    decisions = {
        'recommended': True,
        'issues': [],
        'matched_skills': [],
        'willing_to_join': None,
        'confidence_score': 0.0,
        'recommended_closing': "Thank you — we'll be in touch within 3 business days."
    }

    # Sentiment handling
    label = sentiment_result.get('label', '').lower()
    score = float(sentiment_result.get('score', 0))
    if 'negative' in label or 'neg' in label or score < 0.4:
        decisions['issues'].append('negative_tone')
        decisions['recommended'] = False

    # Skill matching
    for skill in SCREENING_KEYWORDS['skills']:
        if skill in text or skill in ' '.join(keywords).lower():
            decisions['matched_skills'].append(skill)

    # Willingness to join
    for token in SCREENING_KEYWORDS['willing_to_join']:
        if token in text:
            decisions['willing_to_join'] = True
            break
    if decisions['willing_to_join'] is None:
        # If not explicitly present, set to False conservatively
        decisions['willing_to_join'] = False

    # Tone issues: interruptions, anger — naive checks
    if 'angry' in text or 'not happy' in text or 'upset' in text:
        decisions['issues'].append('emotional_distress')
        decisions['recommended'] = False

    # Confidence score heuristic
    decisions['confidence_score'] = min(1.0, 0.2 + 0.2 * len(decisions['matched_skills']) + (0.6 if decisions['willing_to_join'] else 0))

    # Recommend escalation if red flags
    if 'negative_tone' in decisions['issues'] or 'emotional_distress' in decisions['issues']:
        decisions['escalate'] = True
        decisions['recommended_closing'] = "We appreciate your time — our senior recruiter will reach out to discuss further."
    else:
        decisions['escalate'] = False

    # Improve closing line depending on sentiment
    if label.lower().startswith('pos'):
        decisions['recommended_closing'] = "Great speaking with you — expect a call/email from us within 48 hours."

    return decisions


# ---------- Streamlit UI ----------

def streamlit_app():
    st.set_page_config(page_title="Calling Agent - Telephonic Screening", layout="wide")
    st.title("📞 Calling Agent — Telephonic Screening (Demo)")

    st.markdown("This demo simulates a telephonic screening agent. Upload audio or run an outbound call (Twilio credentials required).")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("1) Simulate / Upload call audio")
        uploaded = st.file_uploader("Upload candidate audio (wav/mp3)", type=["wav", "mp3", "m4a", "ogg"])
        run_transcribe = st.button("Transcribe & Analyze uploaded audio")

        if uploaded and run_transcribe:
            st.info("Saving and converting audio...")
            wav_path = save_uploaded_audio(uploaded)
            st.success(f"Saved temp audio: {wav_path}")

            st.info("Transcribing audio...")
            if HAVE_WHISPER:
                result = transcribe_with_whisper(wav_path)
            else:
                result = simple_transcribe_placeholder(wav_path)

            transcript = result.get('text', '')
            st.subheader("Transcript")
            st.write(transcript)

            # Sentiment
            pipe = load_sentiment_pipeline()
            sentiment = get_sentiment(transcript, pipe)
            st.subheader("Sentiment")
            st.json(sentiment)

            # Keywords
            keywords = extract_keywords_yake(transcript)
            st.subheader("Keywords / Key phrases")
            st.write(keywords)

            # Decisions
            decisions = decision_engine(transcript, sentiment, keywords)
            st.subheader("Screening Decisions")
            st.json(decisions)

            # Sample feedback snippet
            st.subheader("Sample Feedback Snippet (for recruiter)")
            feedback = build_feedback_snippet(transcript, sentiment, keywords, decisions)
            st.code(feedback)

    with col2:
        st.header("2) Twilio outbound (example)")
        st.markdown("This section shows example code to place an outbound call via Twilio, which dials a candidate and hits your webhook with the recording URL when done.")
        st.text_area("Twilio example (place credentials in env):", height=220, value=TWILIO_SNIPPET())
        st.markdown("---")
        st.header("3) Webhook for receiving call recordings (Flask)")
        st.text_area("Webhook example (run separately, must be public):", height=220, value=TWILIO_WEBHOOK_SNIPPET())

    st.markdown("---")
    st.markdown("**Notes / Next steps:** Use ngrok to expose webhook in dev. Store transcripts and metadata in a DB. Add speaker diarization for multi-turn analysis.")


def build_feedback_snippet(transcript, sentiment, keywords, decisions) -> str:
    lines = []
    lines.append("Candidate Transcript (excerpt):")
    excerpt = transcript[:300] + ("..." if len(transcript) > 300 else "")
    lines.append(excerpt)
    lines.append("")
    lines.append(f"Sentiment: {sentiment.get('label')} (score={sentiment.get('score')})")
    lines.append(f"Matched skills: {', '.join(decisions.get('matched_skills', [])) or 'None'}")
    lines.append(f"Willing to join: {'Yes' if decisions.get('willing_to_join') else 'No'}")
    lines.append(f"Issues flagged: {', '.join(decisions.get('issues', [])) or 'None'}")
    lines.append(f"Confidence: {decisions.get('confidence_score'):.2f}")
    lines.append("")
    lines.append("Recommended recruiter message:")
    lines.append(decisions.get('recommended_closing'))
    return "\n".join(lines)


# ---------- Twilio example string snippets ----------

def TWILIO_SNIPPET():
    return """
# Example Twilio outbound call (Python)
from twilio.rest import Client

account_sid = os.environ.get('TWILIO_ACCOUNT_SID')
auth_token = os.environ.get('TWILIO_AUTH_TOKEN')
client = Client(account_sid, auth_token)

call = client.calls.create(
    to='+91XXXXXXXXXX',
    from_='+1TWILIONUM',
    url='https://your-public-server/twilio/voice'  # TwiML URL or webhook that returns TwiML
)
print(call.sid)
"""


def TWILIO_WEBHOOK_SNIPPET():
    return """
# Flask webhook example that Twilio can POST to after a call completes (recordingUrl will be sent)
from flask import Flask, request
app = Flask(__name__)

@app.route('/twilio/webhook', methods=['POST'])
def twilio_webhook():
    # Twilio will send RecordingUrl parameter when using <Record> or via REST API
    recording_url = request.form.get('RecordingUrl')
    call_sid = request.form.get('CallSid')
    # You should download the recording, transcribe and process
    # Download code omitted for brevity
    return ('', 204)

if __name__ == '__main__':
    app.run(port=5001)
"""


# ---------- Optional quick Flask webhook runner for demo (not the main Streamlit loop) ----------

def run_webhook_server():
    if not HAVE_TWILIO:
        print("Twilio/Flask not installed. Install 'twilio' and 'flask' to run webhook demo.")
        return

    app = Flask(__name__)

    @app.route('/twilio/voice', methods=['POST', 'GET'])
    def voice():
        # Return simple TwiML to record a call then hangup
        twiml = """<?xml version='1.0' encoding='UTF-8'?><Response><Say>Hello. This is an automated screening. Please answer the following questions after each beep. Your responses may be recorded. Beep.</Say><Record timeout='5' transcribe='false' playBeep='true' maxLength='120' /></Response>"""
        return twiml, 200, {'Content-Type': 'text/xml'}

    @app.route('/twilio/webhook', methods=['POST'])
    def recording_webhook():
        recording_url = request.form.get('RecordingUrl')
        call_sid = request.form.get('CallSid')
        print('Received recording URL:', recording_url)
        # Here you'd download the recording and call your transcription -> analysis pipeline
        return ('', 204)

    app.run(host='0.0.0.0', port=5001)


# ---------- Entry point ----------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-webhook', action='store_true', help='Run Flask webhook demo (separate from Streamlit)')
    args = parser.parse_args()
    if args.run_webhook:
        run_webhook_server()
    else:
        streamlit_app()
