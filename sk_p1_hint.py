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
from random import sample
st.session_state.sk_question = ''
sys_random = random.SystemRandom()
topic_question = [
  {
    "category": "Family",
    "questions": [
      "Do you have any siblings?",
      "Do you have a close relationship with your family members?",
      "Do you spend much time with your family?",
      "Are family traditions important to you?",
      "Who are you closest to in your family?",
    ],
  },
  {
    "category": "Hometown",
    "questions": [
      "What is your hometown like?",
      "Do you often visit your hometown?",
      "What do you like most about your hometown?",
      "Are there any famous landmarks in your hometown?",
      "How has your hometown changed over the years?",
    ],
  },
  {
    "category": "Technology",
    "questions": [
      "How often do you use technology in your daily life?",
      "What is your favorite gadget or app?",
      "Do you think technology has more positive or negative effects on society?",
      "What electronic devices have you bought lately?",
      "What technology do you often use, computers or cell phones?",
    ],
  }
]
question_hint = [
  {
    'question': "How often do you use technology in your daily life?",
    'phrases': [
      { 'phrase': "check my messages", 'meaning': "kiểm tra tin nhắn", 'example': "I check my messages on my phone several times a day." },
      { 'phrase': "use my smartphone", 'meaning': "sử dụng điện thoại thông minh", 'example': "I use my smartphone for various tasks throughout the day." },
      { 'phrase': "browse the internet", 'meaning': "duyệt web", 'example': "I often browse the internet to find information and news." },
      { 'phrase': "send emails", 'meaning': "gửi email", 'example': "I send emails to communicate with colleagues and friends." },
      { 'phrase': "watch videos online", 'meaning': "xem video trực tuyến", 'example': "I enjoy watching videos online during my free time." },
      { 'phrase': "listen to music", 'meaning': "nghe nhạc", 'example': "I listen to music on my phone while commuting or working." },
      { 'phrase': "take photos", 'meaning': "chụp ảnh", 'example': "I frequently take photos with my smartphone to capture moments." },
      { 'phrase': "use social media", 'meaning': "sử dụng mạng xã hội", 'example': "I use social media to connect with friends and share updates." },
      { 'phrase': "set reminders", 'meaning': "đặt nhắc nhở", 'example': "I set reminders on my phone to remember important tasks and appointments." },
      { 'phrase': "play mobile games", 'meaning': "chơi game trên điện thoại", 'example': "I play mobile games for entertainment during breaks." },
      { 'phrase': "check the weather forecast", 'meaning': "kiểm tra dự báo thời tiết", 'example': "I check the weather forecast on my phone before going out." },
      { 'phrase': "use apps for productivity", 'meaning': "sử dụng ứng dụng để tăng cường năng suất", 'example': "I use apps to stay organized and enhance my productivity." },
      { 'phrase': "text friends and family", 'meaning': "nhắn tin cho bạn bè và gia đình", 'example': "I frequently text my friends and family to stay in touch." },
      { 'phrase': "search for recipes online", 'meaning': "tìm kiếm công thức nấu ăn trực tuyến", 'example': "I search for recipes online when planning meals." },
      { 'phrase': "use technology for learning", 'meaning': "sử dụng công nghệ để học", 'example': "I use technology for learning new things and acquiring knowledge." },
    ]  
  },
  {
    'question': "Do you have any siblings?",
    'phrases': [
      { 'phrase': "come from a large family", 'meaning': "đến từ gia đình đông thành viên", 'example': "I come from a large family with four siblings." },
      { 'phrase': "have siblings at home", 'meaning': "có anh chị em ở nhà", 'example': "I have two siblings who still live at home with my parents." },
      { 'phrase': "share my childhood with siblings", 'meaning': "chia sẻ tuổi thơ với anh chị em", 'example': "I shared my childhood with my siblings, and we have many fond memories together." },
      { 'phrase': "grow up with brothers and sisters", 'meaning': "lớn lên cùng anh chị em", 'example': "I grew up with three brothers, and we were very close." },
      { 'phrase': "have siblings of different ages", 'meaning': "có anh chị em ở các độ tuổi khác nhau", 'example': "I have siblings of different ages, ranging from teenagers to adults." },
      { 'phrase': "come from a small family", 'meaning': "đến từ gia đình nhỏ", 'example': "I come from a small family with only one sibling, my sister." },
      { 'phrase': "live with my siblings", 'meaning': "sống cùng anh chị em", 'example': "I currently live with my two siblings in the same house." },
      { 'phrase': "have younger brothers or sisters", 'meaning': "có em gái hoặc em trai", 'example': "Yes, I have two younger brothers who are still in school." },
      { 'phrase': "have an older sister", 'meaning': "có một chị gái", 'example': "I have an older sister who is married and lives in another city." },
      { 'phrase': "grow up as the only child", 'meaning': "lớn lên là con một", 'example': "I grew up as the only child, so I had a lot of attention from my parents." },
      { 'phrase': "have siblings who are twins", 'meaning': "có anh chị em sinh đôi", 'example': "I have siblings who are twins, and they are very close to each other." },
      { 'phrase': "be the youngest in the family", 'meaning': "là em út trong gia đình", 'example': "I am the youngest in the family with two older sisters." },
      { 'phrase': "have step-siblings", 'meaning': "có anh chị em kế", 'example': "I have step-siblings from my parent's second marriage." },
      { 'phrase': "have half-siblings", 'meaning': "có anh chị em cùng cha khác mẹ", 'example': "I have half-siblings from my father's previous marriage." },
    ]
  },
  {
    'question': "Do you have a close relationship with your family members?",
    'phrases': [
      { 'phrase': "spend quality time with", 'meaning': "dành thời gian chất lượng với", 'example': "I make sure to spend quality time with my family on weekends." },
      { 'phrase': "share special moments with", 'meaning': "chia sẻ những khoảnh khắc đặc biệt với", 'example': "I love sharing special moments with my family, like birthdays and holidays." },
      { 'phrase': "have a strong bond with", 'meaning': "có một mối liên kết mạnh mẽ với", 'example': "I have a strong bond with my siblings, and we support each other." },
      { 'phrase': "maintain a close connection with", 'meaning': "duy trì mối liên kết chặt chẽ với", 'example': "Despite the distance, I make an effort to maintain a close connection with my family." },
      { 'phrase': "build a tight-knit relationship with", 'meaning': "xây dựng mối quan hệ chặt chẽ với", 'example': "I aim to build a tight-knit relationship with my extended family as well." },
      { 'phrase': "have a warm relationship with", 'meaning': "có một mối quan hệ ấm áp với", 'example': "I'm fortunate to have a warm relationship with my parents, and we share our thoughts openly." },
      { 'phrase': "create lasting memories with", 'meaning': "tạo những kí ức lâu dài với", 'example': "We often create lasting memories with family outings and gatherings." },
      { 'phrase': "feel a deep connection with", 'meaning': "cảm thấy một mối liên kết sâu sắc với", 'example': "I feel a deep connection with my family, and we understand each other well." },
      { 'phrase': "have a close-knit family", 'meaning': "có một gia đình gắn bó", 'example': "I am fortunate to have a close-knit family that supports and cares for each other." },
      { 'phrase': "nurture a strong family bond", 'meaning': "nuôi dưỡng một mối liên kết mạnh mẽ trong gia đình", 'example': "We actively nurture a strong family bond through communication and shared activities." },
      { 'phrase': "feel a sense of belonging with", 'meaning': "cảm thấy một cảm giác thuộc về với", 'example': "Being with my family, I always feel a sense of belonging and acceptance." },
      { 'phrase': "cultivate a loving relationship with", 'meaning': "nuôi dưỡng một mối quan hệ yêu thương với", 'example': "I strive to cultivate a loving relationship with my family members." },
      { 'phrase': "share a close connection with", 'meaning': "chia sẻ mối liên kết chặt chẽ với", 'example': "I share a close connection with my family through regular communication and visits." },
      { 'phrase': "have a tight bond with", 'meaning': "có một mối liên kết chặt chẽ với", 'example': "I have a tight bond with my siblings, and we always support each other." },
      { 'phrase': "cherish the closeness with", 'meaning': "trân trọng sự gần gũi với", 'example': "I cherish the closeness with my family and make an effort to stay connected." },
    ]
  },
  {
    'question': "Do you spend much time with your family?",
    'phrases': [
      { 'phrase': "enjoy family outings", 'meaning': "thưởng thức những chuyến dã ngoại gia đình", 'example': "I enjoy family outings to parks and beaches on weekends." },
      { 'phrase': "have family dinners", 'meaning': "dùng bữa tối với gia đình", 'example': "We have family dinners together every night to catch up on our day." },
      { 'phrase': "spend weekends with family", 'meaning': "dành cuối tuần với gia đình", 'example': "I often spend weekends with family, doing activities we all enjoy." },
      { 'phrase': "celebrate family events", 'meaning': "kỷ niệm các sự kiện gia đình", 'example': "We celebrate family events like birthdays and anniversaries with joy." },
      { 'phrase': "share family stories", 'meaning': "chia sẻ những câu chuyện gia đình", 'example': "During family gatherings, we share family stories and memories." },
      { 'phrase': "attend family gatherings", 'meaning': "tham gia các buổi tụ tập gia đình", 'example': "I regularly attend family gatherings to stay connected with relatives." },
      { 'phrase': "help with family chores", 'meaning': "giúp đỡ trong công việc nhà của gia đình", 'example': "I help with family chores like cleaning and cooking on weekends." },
      { 'phrase': "visit family members", 'meaning': "thăm họ hàng", 'example': "I make it a point to visit family members, especially during holidays." },
      { 'phrase': "participate in family activities", 'meaning': "tham gia vào các hoạt động gia đình", 'example': "We participate in family activities like board games and movie nights." },
      { 'phrase': "create family traditions", 'meaning': "tạo ra những truyền thống gia đình", 'example': "We create family traditions, like baking together during the holidays." },
      { 'phrase': "spend quality time together", 'meaning': "dành thời gian chất lượng cùng nhau", 'example': "We make an effort to spend quality time together, whether it's playing games or just talking." },
      { 'phrase': "build strong family bonds", 'meaning': "xây dựng mối quan hệ gia đình mạnh mẽ", 'example': "By doing activities together, we build strong family bonds." },
      { 'phrase': "attend family events", 'meaning': "tham gia các sự kiện gia đình", 'example': "I make sure to attend family events, like weddings and reunions, to show my support." },
      { 'phrase': "create lasting family memories", 'meaning': "tạo ra những kí ức gia đình lâu dài", 'example': "Our family vacations help create lasting memories that we cherish." },
      { 'phrase': "have family discussions", 'meaning': "thảo luận với gia đình", 'example': "We have family discussions to share opinions and make decisions together." },
    ]
  },
  {
    'question': "Are family traditions important to you?",
    'phrases': [
      { 'phrase': "family traditions", 'meaning': "những truyền thống gia đình", 'example': "I like family traditions because they are fun and make us happy." },
      { 'phrase': "family customs", 'meaning': "phong tục của gia đình", 'example': "I enjoy family customs because they bring us together." },
      { 'phrase': "family rituals", 'meaning': "nghi lễ trong gia đình", 'example': "I think family rituals are nice as they make our family special." },
      { 'phrase': "family habits", 'meaning': "những thói quen gia đình", 'example': "I love family habits because they are a part of who we are." },
      { 'phrase': "family routines", 'meaning': "thói quen gia đình", 'example': "I find family routines important as they help us every day." },
      { 'phrase': "family ceremonies", 'meaning': "những lễ kỷ niệm của gia đình", 'example': "I enjoy family ceremonies because they make us feel close." },
      { 'phrase': "family conventions", 'meaning': "những phong tục gia đình", 'example': "I value family conventions because they make us unique." },
      { 'phrase': "family practices", 'meaning': "những thói quen gia đình", 'example': "I love family practices because they make us happy." },
    ]
  },
  {
    'question': "Who are you closest to in your family?",
    'phrases': [
      {'phrase': "share a strong bond", 'meaning': "chia sẻ mối liên kết mạnh mẽ", 'example': "I am closest to my younger sister; we share a strong bond and confide in each other about various aspects of our lives."},
      { 'phrase': "rely on for support", 'meaning': "trông cậy vào sự hỗ trợ", 'example': "I rely on my uncle for support; he gives me valuable advice when I need it." },
      { 'phrase': "have a good connection with", 'meaning': "có một mối liên kết tốt với", 'example': "I have a good connection with my cousin; we have similar interests and hobbies." },
      {'phrase': "share a close relationship with", 'meaning': "chia sẻ mối quan hệ gắn bó với", 'example': "I share a close relationship with my mother; her unconditional love and support have been the pillars of strength in my life."},
      {'phrase': "have a tight-knit connection with", 'meaning': "có mối liên kết chặt chẽ với", 'example': "I have a tight-knit connection with my cousin; we grew up together and have a deep understanding of each other's thoughts and feelings."},
      {'phrase': "form a strong attachment to", 'meaning': "tạo ra một liên kết mạnh mẽ với", 'example': "I formed a strong attachment to my aunt; she has been a mentor and a friend, and I often turn to her for advice and guidance."},
      {'phrase': "feel a deep kinship with", 'meaning': "cảm thấy gần gũi sâu sắc với", 'example': "I feel a deep kinship with my older sister; we have a shared history and understanding that goes beyond the typical sibling relationship."},
      {'phrase': "have a special connection with", 'meaning': "có một mối kết nối đặc biệt với", 'example': "I have a special connection with my grandfather; he imparts valuable life lessons, and I cherish the time we spend together."},
      {'phrase': "connect closely with", 'meaning': "kết nối mật thiết với", 'example': "I connect closely with my younger cousin; our age proximity has allowed us to grow up together, creating a strong and enduring connection."},
      {'phrase': "maintain a deep connection with", 'meaning': "duy trì một mối kết nối sâu sắc với", 'example': "I maintain a deep connection with my older brother; he is someone I can rely on for advice, and we share a sense of loyalty and trust."},
      { 'phrase': "get along well with", 'meaning': "hòa thuận tốt với", 'example': "I get along well with my younger sister; we share a lot of interests." },
      { 'phrase': "spend a lot of time with", 'meaning': "dành nhiều thời gian với", 'example': "I spend a lot of time with my younger sister, helping her with homework." },
      { 'phrase': "feel close to", 'meaning': "cảm thấy gần gũi với", 'example': "I feel close to my aunt; we often share our thoughts and experiences." },
      { 'phrase': "feel emotionally connected to", 'meaning': "cảm thấy kết nối cảm xúc với", 'example': "I feel emotionally connected to my cousin; we understand each other's feelings without saying much." },
      { 'phrase': "be tight with", 'meaning': "thân thiết với", 'example': "I am tight with my sister; we share our secrets and support each other." },
      { 'phrase': "maintain a close connection with", 'meaning': "duy trì mối kết nối chặt chẽ với", 'example': "I maintain a close connection with my grandfather; he tells me stories from the past." },
    ]
  },
  {
    'question': "Do you often visit your hometown?",
    'phrases': [
      {'phrase': "stay connected with roots", 'meaning': "kết nối với cội nguồn", 'example': "Visiting my hometown allows me to stay connected with my cultural roots."},
      {'phrase': "reconnect with old friends", 'meaning': "kết nối lại với bạn bè cũ", 'example': "I make it a point to visit my hometown to reconnect with old friends and classmates."},
      {'phrase': "explore familiar surroundings", 'meaning': "khám phá môi trường quen thuộc", 'example': "During my visits, I take time to explore the familiar surroundings of my hometown."},
      {'phrase': "experience local traditions", 'meaning': "trải nghiệm những truyền thống địa phương", 'example': "I love visiting during festivals to experience local traditions and celebrations."},
      {'phrase': "spend quality time with family", 'meaning': "dành thời gian chất lượng với gia đình", 'example': "Visiting my hometown allows me to spend quality time with my family members."},
      {'phrase': "revisit favorite childhood spots", 'meaning': "quay lại những địa điểm thơ ấu ưa thích", 'example': "I always make a point to revisit my favorite childhood spots whenever I go back to my hometown."},
      {'phrase': "strengthen family bonds", 'meaning': "củng cố mối quan hệ gia đình", 'example': "Frequent visits help strengthen family bonds and maintain a close connection with relatives."},
      {'phrase': "recharge my emotional batteries", 'meaning': "nạp lại năng lượng cảm xúc", 'example': "Visiting my hometown acts as a way to recharge my emotional batteries and find inner peace."},
      {'phrase': "celebrate family milestones", 'meaning': "kỷ niệm những cột mốc của gia đình", 'example': "We often gather in our hometown to celebrate important family milestones and achievements."},
      {'phrase': "explore historical landmarks", 'meaning': "khám phá các địa danh lịch sử", 'example': "One of the highlights of visiting my hometown is exploring historical landmarks and monuments."},
      {'phrase': "relive precious childhood moments", 'meaning': "hồi tưởng lại những khoảnh khắc quý giá thời thơ ấu", 'example': "Returning to my hometown allows me to relive precious childhood moments with friends and family."},
      {'phrase': "connect with my roots", 'meaning': "kết nối với cội nguồn", 'example': "Visiting my hometown is a way for me to connect with my cultural roots and heritage."},
      {'phrase': "strengthen ties with childhood friends", 'meaning': "củng cố mối quan hệ với bạn bè thời thơ ấu", 'example': "My hometown visits give me the chance to strengthen ties with my childhood friends."},
      {'phrase': "reconnect with my origins", 'meaning': "kết nối lại với nguồn gốc của mình", 'example': "Visiting my hometown allows me to reconnect with my origins and the essence of who I am."},
      {'phrase': "rediscover hidden gems", 'meaning': "khám phá lại những điểm đẹp ẩn mình", 'example': "Visiting my hometown allows me to rediscover hidden gems and places I may have overlooked."},
      {'phrase': "create lasting memories with family", 'meaning': "tạo ra những kỷ niệm lâu dài với gia đình", 'example': "Every visit becomes an opportunity to create lasting memories with my family."},
    ]
  },
  {
    'question': "What do you like most about your hometown?",
    'phrases': [
      { 'phrase': "love the friendly people", 'meaning': "yêu những người thân thiện", 'example': "I love the friendly people in my hometown who always greet each other warmly." },
      { 'phrase': "enjoy the local cuisine", 'meaning': "thưởng thức ẩm thực địa phương", 'example': "I enjoy the local cuisine in my hometown, especially the traditional dishes." },
      { 'phrase': "like the peaceful atmosphere", 'meaning': "thích không khí yên bình", 'example': "I like the peaceful atmosphere in my hometown, away from the hustle and bustle of the city." },
      { 'phrase': "appreciate the natural beauty", 'meaning': "đánh giá cao vẻ đẹp tự nhiên", 'example': "I appreciate the natural beauty of my hometown, with its scenic landscapes and greenery." },
      { 'phrase': "admire the historical sites", 'meaning': "ngưỡng mộ các di tích lịch sử", 'example': "I admire the historical sites in my hometown that reflect its rich cultural heritage." },
      { 'phrase': "value the sense of community", 'meaning': "trân trọng tinh thần cộng đồng", 'example': "I value the strong sense of community in my hometown, where everyone knows and supports each other." },
      { 'phrase': "like the local traditions", 'meaning': "thích những truyền thống địa phương", 'example': "I like the local traditions in my hometown, especially the festivals and celebrations." },
      { 'phrase': "enjoy the close-knit neighborhood", 'meaning': "thưởng thức không khí thân thiện của khu phố", 'example': "I enjoy the close-knit neighborhood atmosphere in my hometown, where neighbors are like extended family." },
      { 'phrase': "love the cultural diversity", 'meaning': "yêu sự đa dạng văn hóa", 'example': "I love the cultural diversity of my hometown, with people from various backgrounds living harmoniously." },
      { 'phrase': "appreciate the safety", 'meaning': "đánh giá cao tính an toàn", 'example': "I appreciate the safety of my hometown, making it a secure place to live." },
      { 'phrase': "like the local markets", 'meaning': "thích những khu chợ địa phương", 'example': "I like the local markets in my hometown, where you can find fresh produce and unique items." },
      { 'phrase': "enjoy the traditional festivals", 'meaning': "thưởng thức các lễ hội truyền thống", 'example': "I enjoy the traditional festivals in my hometown, which bring the community together in celebration." },
      { 'phrase': "value the sense of familiarity", 'meaning': "trân trọng cảm giác quen thuộc", 'example': "I value the sense of familiarity in my hometown, where I know the streets and people well." },
      { 'phrase': "love the local parks", 'meaning': "yêu các công viên địa phương", 'example': "I love the local parks in my hometown, where I can relax and enjoy nature." },
      { 'phrase': "appreciate the slower pace of life", 'meaning': "đánh giá cao nhịp sống chậm rãi", 'example': "I appreciate the slower pace of life in my hometown, allowing for a more relaxed lifestyle." },

    ]
  },
  {
    'question': "What is your hometown like?",
    'phrases': [
      { 'phrase': "small and peaceful", 'meaning': "nhỏ và yên bình", 'example': "My hometown is small and peaceful, with friendly neighbors." },
      { 'phrase': "located by the sea", 'meaning': "nằm ven biển", 'example': "My hometown is located by the sea, offering beautiful beach views." },
      { 'phrase': "surrounded by mountains", 'meaning': "được bao quanh bởi núi", 'example': "My hometown is surrounded by mountains, creating a scenic landscape." },
      { 'phrase': "known for historic sites", 'meaning': "nổi tiếng với các địa điểm lịch sử", 'example': "My hometown is known for historic sites that attract tourists." },
      { 'phrase': "rich in cultural heritage", 'meaning': "phong phú về di sản văn hóa", 'example': "My hometown is rich in cultural heritage, with traditional festivals and customs." },
      { 'phrase': "famous for local cuisine", 'meaning': "nổi tiếng với ẩm thực địa phương", 'example': "My hometown is famous for its local cuisine, especially a signature dish." },
      { 'phrase': "bustling with markets", 'meaning': "náo nhiệt với các chợ", 'example': "My hometown is bustling with markets where you can find fresh produce." },
      { 'phrase': "green and picturesque", 'meaning': "xanh và đẹp như tranh", 'example': "My hometown is green and picturesque, with parks and gardens." },
      { 'phrase': "full of friendly people", 'meaning': "đầy người thân thiện", 'example': "My hometown is full of friendly people who make you feel welcome." },
      { 'phrase': "vibrant nightlife", 'meaning': "đời sống về đêm sôi động", 'example': "My hometown has a vibrant nightlife with various entertainment options." },
      { 'phrase': "center of local traditions", 'meaning': "trung tâm của các truyền thống địa phương", 'example': "My hometown is the center of local traditions, hosting cultural events." },
      { 'phrase': "known for its parks", 'meaning': "nổi tiếng với các công viên", 'example': "My hometown is known for its parks, providing recreational spaces for residents." },
      { 'phrase': "accessible by public transport", 'meaning': "tiện lợi với phương tiện giao thông công cộng", 'example': "My hometown is accessible by public transport, making it easy to get around." },
      { 'phrase': "close-knit community", 'meaning': "cộng đồng gắn bó", 'example': "My hometown has a close-knit community where everyone knows each other." },
      { 'phrase': "frequent cultural events", 'meaning': "các sự kiện văn hóa thường xuyên", 'example': "My hometown hosts frequent cultural events, showcasing local talent." },
    ]
  },
  {
    'question': "Are there any famous landmarks in your hometown?",
    'phrases': [
      { 'phrase': "have historical buildings", 'meaning': "có các công trình lịch sử", 'example': "My hometown has historical buildings that tell stories of the past." },
      { 'phrase': "feature well-known monuments", 'meaning': "có các di tích nổi tiếng", 'example': "There are well-known monuments in my hometown that attract tourists." },
      { 'phrase': "boast iconic structures", 'meaning': "tự hào với các công trình biểu tượng", 'example': "My hometown boasts iconic structures that symbolize its identity." },
      { 'phrase': "include famous landmarks", 'meaning': "bao gồm các điểm địa danh nổi tiếng", 'example': "The city includes famous landmarks that are recognized nationally." },
      { 'phrase': "contain recognized sites", 'meaning': "bao gồm những địa điểm được công nhận", 'example': "My hometown contains recognized sites that have cultural significance." },
      { 'phrase': "well-known places", 'meaning': "những địa điểm nổi tiếng", 'example': "There are well-known places in my hometown where people often gather." },
      { 'phrase': "feature popular attractions", 'meaning': "đặc trưng với các điểm thu hút phổ biến", 'example': "The town features popular attractions that draw visitors throughout the year." },
      { 'phrase': "boast famous landmarks", 'meaning': "tự hào với các điểm địa danh nổi tiếng", 'example': "Our hometown boasts famous landmarks that reflect its cultural heritage." },
      { 'phrase': "include renowned sites", 'meaning': "bao gồm các địa điểm nổi tiếng", 'example': "The city includes renowned sites that are often featured in travel guides." },
      { 'phrase': "have iconic places", 'meaning': "có những địa điểm biểu tượng", 'example': "Our town has iconic places that symbolize its uniqueness." },
      { 'phrase': "contain well-known landmarks", 'meaning': "bao gồm các điểm địa danh nổi tiếng", 'example': "The region contains well-known landmarks that are part of our heritage." },
      { 'phrase': "feature famous sites", 'meaning': "đặc sắc với các địa điểm nổi tiếng", 'example': "Our hometown features famous sites that attract visitors from far and wide." },
      { 'phrase': "boast recognized structures", 'meaning': "tự hào với các công trình được công nhận", 'example': "The city boasts recognized structures that showcase its architectural history." },
      { 'phrase': "popular landmarks", 'meaning': "các điểm địa danh phổ biến", 'example': "There are popular landmarks in my hometown that people often visit for sightseeing." },
      { 'phrase': "have well-known buildings", 'meaning': "có các công trình nổi tiếng", 'example': "Our hometown has well-known buildings that are considered architectural treasures." },
    ]
  },
  {
    'question': "How has your hometown changed over the years?",
    'phrases': [
      { 'phrase': "see new buildings", 'meaning': "nhìn thấy các tòa nhà mới", 'example': "Over the years, I've seen new buildings being constructed in my hometown." },
      { 'phrase': "increased population", 'meaning': "dân số tăng lên", 'example': "One significant change is the increased population in my hometown." },
      { 'phrase': "observe improved infrastructure", 'meaning': "quan sát cơ sở hạ tầng được cải thiện", 'example': "I've observed improved infrastructure with better roads and facilities." },
      { 'phrase': "witness more businesses", 'meaning': "chứng kiến sự gia tăng của các doanh nghiệp", 'example': "I've witnessed the emergence of more businesses in the area." },
      { 'phrase': "experience better public transport", 'meaning': "trải qua sự cải thiện của giao thông công cộng", 'example': "One positive change is the experience of better public transport options." },
      { 'phrase': "new parks and green spaces", 'meaning': "các công viên và không gian xanh mới", 'example': "New parks and green spaces have been added to enhance the environment." },
      { 'phrase': "notice upgraded facilities", 'meaning': "nhận thấy cơ sở vật chất được nâng cấp", 'example': "I've noticed upgraded facilities, such as schools and hospitals." },
      { 'phrase': "modernized transportation", 'meaning': "giao thông được hiện đại hóa", 'example': "There has been a noticeable modernization of transportation in my hometown." },
      { 'phrase': "improve technology access", 'meaning': "cải thiện trong việc tiếp cận công nghệ", 'example': "People in my hometown now have improved access to technology." },
      { 'phrase': "cultural events", 'meaning': "các sự kiện văn hóa", 'example': "There has been an increase in the number of cultural events and festivals." },
      { 'phrase': "local businesses", 'meaning': "các doanh nghiệp địa phương", 'example': "Local businesses have undergone changes, adapting to new trends." },
      { 'phrase': "educational facilities", 'meaning': "cơ sở vật chất cho giáo dục", 'example': "The educational facilities in my hometown have noticeably improved." },
      { 'phrase': "observe enhanced community services", 'meaning': "quan sát sự cải thiện trong dịch vụ cộng đồng", 'example': "There has been an observable enhancement in community services." },
      { 'phrase': "notice improved public services", 'meaning': "nhận thấy cải thiện trong các dịch vụ công cộng", 'example': "I've noticed improved public services, making daily life more convenient for residents." },
      { 'phrase': "witness a rise in tourism", 'meaning': "chứng kiến sự tăng lên trong du lịch", 'example': "One significant change is witnessing a rise in tourism, attracting more visitors to my hometown." },
    ]
  },
  {
    'question': "What is your favorite gadget or app?",
    'phrases': [
      { 'phrase': "appreciate my phone", 'meaning': "đánh giá cao chiếc điện thoại của mình", 'example': "I appreciate my phone because it keeps me connected with friends and family." },
      { 'phrase': "enjoy my tablet", 'meaning': "thích sử dụng máy tính bảng của mình", 'example': "I thoroughly enjoy my tablet, especially for playing games and watching videos." },
      { 'phrase': "value my computer", 'meaning': "đánh giá cao máy tính của mình", 'example': "I value my computer as it's essential for both work and school tasks." },
      { 'phrase': "treasure my smartwatch", 'meaning': "trân trọng chiếc đồng hồ thông minh của mình", 'example': "I treasure my smartwatch because it helps me monitor my daily activities." },
      { 'phrase': "find my fitness tracker beneficial", 'meaning': "thấy rằng dụng cụ theo dõi sức khỏe của mình mang lại lợi ích", 'example': "I find my fitness tracker beneficial for keeping track of my exercise routine." },
      { 'phrase': "relish using navigation apps", 'meaning': "thích việc sử dụng ứng dụng định vị", 'example': "I relish using navigation apps to easily find my way around new places." },
      { 'phrase': "hold my e-reader in high regard", 'meaning': "đánh giá cao máy đọc sách điện tử của mình", 'example': "I hold my e-reader in high regard as it allows me to carry many books conveniently." },
      { 'phrase': "cherish my camera", 'meaning': "trân trọng việc sử dụng máy ảnh của mình", 'example': "I cherish my camera for capturing precious moments with excellent clarity." },
      { 'phrase': "value my headphones", 'meaning': "đánh giá cao tai nghe của mình", 'example': "I value my headphones for delivering good sound quality and a great listening experience." },
      { 'phrase': "take pleasure in my gaming console", 'meaning': "hưởng thụ việc sử dụng máy chơi game của mình", 'example': "I take pleasure in my gaming console, especially for playing exciting video games." },
      { 'phrase': "hold my alarm clock in high esteem", 'meaning': "đánh giá cao đồng hồ báo thức của mình", 'example': "I hold my alarm clock in high esteem as it wakes me up gently every morning." },
      { 'phrase': "appreciate the functionality of my calculator", 'meaning': "đánh giá cao tính năng của máy tính của mình", 'example': "I appreciate the functionality of my calculator, especially for math homework." },
      { 'phrase': "find satisfaction in my smart home device", 'meaning': "cảm thấy hài lòng khi sử dụng thiết bị nhà thông minh của mình", 'example': "I find satisfaction in my smart home device for controlling lights and temperature." }
        ]
  },
  {
    'question': "Do you think technology has more positive or negative effects on society?",
    'phrases': [
      { 'phrase': "helps people talk to each other", 'meaning': "giúp mọi người nói chuyện với nhau", 'example': "Phones help people talk to each other easily." },
      { 'phrase': "makes work easier", 'meaning': "làm cho công việc dễ dàng hơn", 'example': "Computers make work easier by doing calculations for us." },
      { 'phrase': "gives quick access to information", 'meaning': "đưa ra thông tin nhanh chóng", 'example': "The internet gives quick access to information on many topics." },
      { 'phrase': "connects friends from far away", 'meaning': "kết nối bạn bè ở xa", 'example': "Social media connects friends from far away." },
      { 'phrase': "helps students learn online", 'meaning': "giúp học sinh học trực tuyến", 'example': "Computers help students learn online with videos and lessons." },
      { 'phrase': "provides fun games and videos", 'meaning': "cung cấp trò chơi và video vui nhộn", 'example': "Tablets provide fun games and videos for entertainment." },
      { 'phrase': "makes everyday tasks easier", 'meaning': "làm cho công việc hàng ngày dễ dàng hơn", 'example': "Smartphones make everyday tasks easier, like setting alarms." },
      { 'phrase': "gives information to everyone", 'meaning': "đưa thông tin đến mọi người", 'example': "TV gives information to everyone about news and events." },
      { 'phrase': "helps people work from home", 'meaning': "giúp mọi người làm việc từ nhà", 'example': "Laptops help people work from home on their computers." },
      { 'phrase': "makes it easy to see a doctor online", 'meaning': "làm cho việc thăm khám với bác sĩ trực tuyến dễ dàng", 'example': "Phones make it easy to see a doctor online for advice." },
      { 'phrase': "saves time with automatic tasks", 'meaning': "tiết kiệm thời gian với chế độ tự động", 'example': "Washing machines save time with automatic washing." },
      { 'phrase': "lets people be creative with pictures", 'meaning': "cho phép mọi người sáng tạo với hình ảnh", 'example': "Cameras let people be creative with pictures they take." },
      { 'phrase': "gives answers quickly", 'meaning': "đưa ra câu trả lời nhanh chóng", 'example': "Search engines give answers quickly when you have a question." },
      { 'phrase': "helps people talk without being together", 'meaning': "giúp mọi người nói chuyện mà không cần gặp mặt", 'example': "Phones help people talk without being together in person." },
      { 'phrase': "reminds people to care for the Earth", 'meaning': "nhắc nhở mọi người quan tâm Trái Đất", 'example': "Apps remind people to care for the Earth by saving energy." }
    ]
  },
  {
    'question': "What electronic devices have you bought lately?",
    'phrases': [
      { 'phrase': "purchase a new smartphone", 'meaning': "mua điện thoại mới", 'example': "I recently purchased a new smartphone with better camera features." },
      { 'phrase': "buy a laptop", 'meaning': "mua laptop", 'example': "I bought a laptop for my online classes and work tasks." },
      { 'phrase': "get a tablet", 'meaning': "sở hữu máy tính bảng", 'example': "I got a tablet for reading e-books and watching videos on the go." },
      { 'phrase': "acquire a smartwatch", 'meaning': "mua đồng hồ thông minh", 'example': "I recently acquired a smartwatch to track my fitness activities." },
      { 'phrase': "obtain a new camera", 'meaning': "sở hữu máy ảnh mới", 'example': "I obtained a new camera to capture high-quality photos during my travels." },
      { 'phrase': "invest in a fitness tracker", 'meaning': "đầu tư vào dụng cụ theo dõi sức khỏe", 'example': "I recently invested in a fitness tracker to monitor my daily activity levels." },
      { 'phrase': "buy wireless earbuds", 'meaning': "mua tai nghe không dây", 'example': "I bought wireless earbuds for a more convenient and tangle-free listening experience." },
      { 'phrase': "get a new television", 'meaning': "sở hữu tivi mới", 'example': "I recently got a new television for better viewing quality." },
      { 'phrase': "buy a gaming console", 'meaning': "mua máy chơi game", 'example': "I purchased a gaming console to enjoy playing video games with friends." },
      { 'phrase': "acquire a portable charger", 'meaning': "mua sạc di động", 'example': "I acquired a portable charger to keep my devices charged while traveling." },
      { 'phrase': "invest in a new printer", 'meaning': "đầu tư vào máy in mới", 'example': "I recently invested in a new printer for my home office needs." },
      { 'phrase': "buy a smart home device", 'meaning': "mua thiết bị nhà thông minh", 'example': "I bought a smart home device to control lights and appliances with my phone." },
      { 'phrase': "get a new keyboard and mouse", 'meaning': "sở hữu bàn phím và chuột mới", 'example': "I got a new keyboard and mouse to enhance my computer setup." },
      { 'phrase': "buy a new router", 'meaning': "mua router mới", 'example': "I recently bought a new router to improve my internet connection at home." },
      { 'phrase': "acquire a power bank", 'meaning': "mua sạc dự phòng", 'example': "I acquired a power bank for emergencies to charge my phone on the go." }
    ]
  },
  {
    'question': "What technology do you often use, computers or cell phones?",
    'phrases': [
      { 'phrase': "make calls", 'meaning': "gọi điện thoại", 'example': "I often use my cell phone to make calls to my friends and family." },
      { 'phrase': "send text messages", 'meaning': "gửi tin nhắn văn bản", 'example': "I use my smartphone to send text messages to communicate with others." },
      { 'phrase': "take photos", 'meaning': "chụp ảnh", 'example': "I love using the camera on my phone to take photos of special moments." },
      { 'phrase': "check emails", 'meaning': "kiểm tra email", 'example': "I use my computer to check emails for work and personal communication." },
      { 'phrase': "browse the internet", 'meaning': "duyệt web", 'example': "I often use my computer to browse the internet for information and entertainment." },
      { 'phrase': "play games", 'meaning': "chơi game", 'example': "I enjoy playing games on my smartphone during my free time." },
      { 'phrase': "use social media", 'meaning': "sử dụng mạng xã hội", 'example': "I stay connected with friends by using social media apps on my phone." },
      { 'phrase': "set alarms", 'meaning': "đặt báo thức", 'example': "I rely on my phone to set alarms to wake me up in the morning." },
      { 'phrase': "listen to music", 'meaning': "nghe nhạc", 'example': "I use my phone to listen to music while I'm commuting or exercising." },
      { 'phrase': "watch videos", 'meaning': "xem video", 'example': "I often watch videos on my computer for entertainment and learning purposes." },
      { 'phrase': "send photos", 'meaning': "gửi ảnh", 'example': "I use my phone to send photos to my friends, especially when we can't meet in person." },
      { 'phrase': "check the weather", 'meaning': "kiểm tra thời tiết", 'example': "Before going out, I check the weather forecast on my phone to plan my day." },
      { 'phrase': "make video calls", 'meaning': "thực hiện cuộc gọi video", 'example': "I connect with my family through video calls on my phone when we are apart." },
      { 'phrase': "write notes", 'meaning': "viết ghi chú", 'example': "I use both my computer and phone to write down important notes and reminders." },
      { 'phrase': "use the calculator", 'meaning': "sử dụng máy tính", 'example': "When I need to calculate something quickly, I use the calculator on my phone." },
      { 'phrase': "check the time", 'meaning': "kiểm tra giờ", 'example': "I check the time on my phone regularly, especially when I don't have a watch." },
      { 'phrase': "set reminders", 'meaning': "đặt nhắc nhở", 'example': "I set reminders on my phone to remember important appointments and tasks." },
      { 'phrase': "read e-books", 'meaning': "đọc sách điện tử", 'example': "I use my tablet to read e-books, especially before going to bed." },
      { 'phrase': "charge the battery", 'meaning': "sạc pin", 'example': "Every night, I charge the battery of my phone to ensure it's ready for the next day." },
      { 'phrase': "send voice messages", 'meaning': "gửi tin nhắn thoại", 'example': "Instead of typing, I often send voice messages on my phone for a more personal touch." },
      { 'phrase': "use navigation apps", 'meaning': "sử dụng ứng dụng định vị", 'example': "I rely on navigation apps on my phone to find directions when I'm in a new place." },
      { 'phrase': "use a password", 'meaning': "sử dụng mật khẩu", 'example': "To protect my information, I always use a password on my computer and phone." },
      { 'phrase': "update apps", 'meaning': "cập nhật ứng dụng", 'example': "I regularly update the apps on my phone to ensure they function properly." },
      { 'phrase': "customize settings", 'meaning': "tùy chỉnh cài đặt", 'example': "I like to customize the settings on my phone to suit my preferences and needs." }
        ]
  },
];
st.title('SPEAKING PART 1')
topics = [topic["category"] for topic in topic_question]
def choose_question(value):
    for top_quest in topic_question:
        if value == top_quest["category"]:
            questions = top_quest["questions"] 
    st.session_state.sk_question = sys_random.choice(questions)
    if st.button('Change question')or 'sk_question'=='':
        st.session_state.sk_question = sys_random.choice(questions)
    return st.session_state.sk_question
def choose_hints(question):
    for quest in question_hint:
        if question == quest['question']:
            phrases = quest["phrases"]
            list_phrases = sample(phrases, k=3)
            for ph in list_phrases:
                st.markdown(f"""
                        **Hint: :green[{ph.get("phrase")}]** (:orange[{ph.get("meaning")}])  
                        **Example**: *{ph.get("example")}*  
                        """)

topic = st.selectbox('**Choose a topic**', options=topics)
question = choose_question(topic)
st.markdown(f"### {st.session_state.sk_question}")
if st.button('Gợi ý'):
    choose_hints(question)

