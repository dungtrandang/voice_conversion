import streamlit as st
import numpy as np
import whisper
from tempfile import NamedTemporaryFile
from gtts import gTTS
from io import BytesIO
import base64
from audio_recorder_streamlit import audio_recorder
from ai_corrector import correctness

def auto_display_audio(audio_obj, caption, audio_type):
    audio_base64 = base64.b64encode(audio_obj.read()).decode('utf-8')
    html_code = f'<audio controls autoplay><source src="data:audio/wav;base64,{audio_base64}" type="audio/{audio_type}"></audio>'
    st.caption(caption)
    st.markdown(html_code, unsafe_allow_html=True)
st.title('Voice Conversion')

audio_type = st.sidebar.radio(
    "Choose audio type",
    [ "Recorder", "File"],
    captions = ["Record audio.", "Upload audio file."])

choose_ai_help = st.radio(
            "Need AI's revision?",
            ["I can handle it", "Yes, please"])

col1, col2 = st.columns(spec=[0.45,0.55], gap="large")

with col1:
    if audio_type == "Recorder":
        st.subheader('Start recording')
        audio = audio_recorder(text="Click to record", icon_size="2x", recording_color="#de3e2c", neutral_color="#6aa36f", pause_threshold=30)
    elif audio_type == "File":
        st.subheader('Upload an audio file')
        audio = st.file_uploader(" ", type=["mp3", "wav", "ogg"])
        
    if audio is not None:
        if audio_type == "Recorder":
            audio = BytesIO(audio)
        if st.button("Play your original voice", type="secondary", use_container_width=True):
            auto_display_audio(audio, "Your original audio","wav")
        # st.audio(audio, format='audio/mp3', start_time=0)
        
        st.write("") #add space before transcribed text
        with NamedTemporaryFile(suffix="mp3") as temp:
            temp.write(audio.getvalue())
            temp.seek(0)

            model = whisper.load_model("base")
            result = model.transcribe(temp.name, fp16=False)
            
            text = result["text"]
            st.caption("Transcribed Text")
            st.write(text)

def display_audio(text_to_display, caption):
    mp3_fp = BytesIO()
    text_to_display.write_to_fp(mp3_fp)
    mp3_fp.seek(0)
    # st.audio(mp3_fp, format='audio/mp3', start_time=0)
    auto_display_audio(mp3_fp,caption,"mp3")
    # return mp3_fp

def ai_help(transcribed_text):
    st.subheader('You can speak better with:')
    refined_output = correctness(transcribed_text)
    st.write(refined_output[0])
    tts_ai = gTTS(text=refined_output[0], lang="en", tld="com")
    if st.button(key="Original", label = "Play", type="secondary"):
        display_audio(tts_ai, "AI voice for the refined text")
    st.caption("Explaination")
    st.write(refined_output[1])

with col2:
    if audio is not None and text!="":
        st.subheader('Transcribed Audio')
        tts = gTTS(text=text, lang="en", tld="com")
        if st.button(key="Refined", label="Play", type="secondary"):
            display_audio(tts, "AI voice for the original text")
        
        if choose_ai_help == "Yes, please":
            ai_help(text)
        
    elif audio is not None and text=="":
        st.write("Something went wrong! Please check your microphone or **upload the audio file** or **record** again.")