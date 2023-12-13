import streamlit as st
import numpy as np
import whisper
from tempfile import NamedTemporaryFile
from gtts import gTTS
from io import BytesIO
import base64
from audio_recorder_streamlit import audio_recorder
from ai_corrector import correctness, question, hint

st.title('SPEAKING PART 1')
level = st.selectbox("Select your level", options =[
    "CERF A1",
    "CERF A2",
    "CERF B1",
    "CERF B2",
    "CERF C1",
    "CERF C2",
])
sk_question = "Do you like shopping" #question()
st.markdown(f"### {sk_question}")

audio = audio_recorder(text="Ask for help", icon_size="2x", recording_color="#de3e2c", neutral_color="#6aa36f", pause_threshold=30)
if audio is not None:
    audio = BytesIO(audio)
    with NamedTemporaryFile(suffix="mp3") as temp:
        temp.write(audio.getvalue())
        temp.seek(0)

        model = whisper.load_model("base")
        result = model.transcribe(temp.name, language='vi', fp16=False)
        
        
        # mel = whisper.log_mel_spectrogram(temp.name).to(model.device)
        # _, probs = model.detect_language(mel)

        # # decode the audio
        # options = whisper.DecodingOptions()
        # result = whisper.decode(model, mel, options)
        text = result["text"]
        st.caption("Your request")
        st.write(text)
else:
    text=st.text_input("",placeholder='Nhập câu hỏi của bạn...', label_visibility="collapsed")
    st.caption("Your request")
    st.write(text)
if text:
    hints = hint(level, sk_question, text).get("hints")
    for hint in hints:
        st.markdown(f"""
                    **Hint: {hint.get("phrase")}**  
                    Meaning: {hint.get("meaning")}  
                    *Example: {hint.get("example")}*  
                    """)