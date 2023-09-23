import streamlit as st
import numpy as np
import whisper
from tempfile import NamedTemporaryFile
from gtts import gTTS
from io import BytesIO
from audio_recorder_streamlit import audio_recorder

st.title('Voice Conversion')

audio_type = st.sidebar.radio(
    "Choose audio type",
    ["File", "Recorder"],
    captions = ["Upload audio file.", "Record audio."])

col1, col2 = st.columns(spec=[0.63,0.37], gap="large")

with col1:
    
    if audio_type == "File":
        st.subheader('Upload an audio file')
        audio = st.file_uploader(" ", type=["mp3", "wav", "ogg"])
    elif audio_type == "Recorder":
        st.subheader('Start recording')
        audio = audio_recorder(text="Click to record",icon_size="2x", recording_color="#de3e2c", neutral_color="#6aa36f", pause_threshold = 30)
        
    if audio is not None:
        if audio_type == "Recorder":
            audio = BytesIO(audio)
        st.caption("Your original audio")
        st.audio(audio, format='audio/mp3')
        
        st.write("") #add space before transcribed text
        with NamedTemporaryFile(suffix="mp3") as temp:
            temp.write(audio.getvalue())
            temp.seek(0)

            model = whisper.load_model("base")
            result = model.transcribe(temp.name, fp16=False)
            
            text = result["text"]

with col2:
    if audio is not None:
        st.subheader('Transcribed Audio')
        st.caption("Transcribed Text")
        st.write(text)
        mp3_fp = BytesIO()
        tts = gTTS(text=text, lang="en", tld="com")
        tts.write_to_fp(mp3_fp)
        st.audio(mp3_fp, format='audio/mp3')