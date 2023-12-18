import streamlit as st
import numpy as np
import whisper
from tempfile import NamedTemporaryFile
from gtts import gTTS
from io import BytesIO
import base64
from audio_recorder_streamlit import audio_recorder
from ai_corrector import correctness, question, hint
import random
st.title('SPEAKING PART 1')
level = st.selectbox("Select your level", options =[
    "CERF A1",
    "CERF A2",
    "CERF B1",
    "CERF B2",
    "CERF C1",
    "CERF C2",
])
questions = [
    "What is your favorite type of food?",
    "How often do you watch TV or movies?",
    "Can you describe a person who you admire?",
    "Do you enjoy cooking? Why or why not?",
    "What are the advantages of living in a big city?",
    "How do you usually celebrate your birthday?",
    "What is your favorite season and why?",
    "How often do you travel? Where was your last trip?",
    "Can you talk about a book you have recently read?",
    "Describe a skill you would like to learn in the future."
    "What is your favorite hobby?",
    "How often do you do that hobby?",
    "What do you like most about your job/studies?",
    "Can you describe your hometown?",
    "What type of accommodation do you live in?",
    "What is your favorite form of entertainment?",
    "How do you usually spend your weekends?"
]
if st.button("Next question") or 'sk_question' not in st.session_state :
    st.session_state.sk_question = random.choice(questions)
st.markdown(f"### {st.session_state.sk_question}")


text=st.text_input("Nhập ý tưởng của bạn",placeholder='Nhập ý tưởng của bạn...', label_visibility="collapsed")
    
if text:
    st.write(f"Yêu cầu của bạn: ***{text}***")
    hints = hint(level, st.session_state.sk_question, text).get("hints")
    if hints:
        for hint in hints:
            st.markdown(f"""
                        **Hint: {hint.get("phrase")}** ({hint.get("meaning")})  
                        *Example: {hint.get("example")}*  
                        """)
    else:
        st.write(""":orange[AI tạm thời chưa sẵn sàng hoặc không có gợi ý cho câu hỏi/yêu cầu này.]""")
        st.write(""":orange[Vui lòng thử lại!]""")












# audio = audio_recorder(text="Ask for help", icon_size="2x", recording_color="#de3e2c", neutral_color="#6aa36f", pause_threshold=30)
# if audio is not None:
#     audio = BytesIO(audio)
#     with NamedTemporaryFile(suffix="mp3") as temp:
#         temp.write(audio.getvalue())
#         temp.seek(0)

#         model = whisper.load_model("base")
#         result = model.transcribe(temp.name, language='vi', fp16=False)
        
        
#         # mel = whisper.log_mel_spectrogram(temp.name).to(model.device)
#         # _, probs = model.detect_language(mel)

#         # # decode the audio
#         # options = whisper.DecodingOptions()
#         # result = whisper.decode(model, mel, options)
#         text = result["text"]