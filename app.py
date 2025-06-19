# app.py
import streamlit as st
import dashscope
from dashscope import Generation
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import re
from collections import deque
from rel import is_semantically_relevant  # <- import fungsi similarity check
import os
from dotenv import load_dotenv

load_dotenv()  # load file .env

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
dashscope.api_key = DASHSCOPE_API_KEY
dashscope.base_http_api_url = 'https://dashscope-intl.aliyuncs.com/api/v1'

st.set_page_config(page_title="💬📝 English Chat + Grammar Fix", layout="wide")
st.title("📱 AI Grammar Coach & Conversation 📱")

# State inisialisasi
if 'messages' not in st.session_state:
    st.session_state.messages = [
        {'role': 'system', 'content': '''You are an English teacher and chatbot. Your task is to:
1. FIRST, identify and correct any grammatical errors in the user's sentence (provide the corrected version)
2. THEN, engage in conversation about the topic
3. Always format your response as: "[CORRECTION] <corrected sentence>\n\n[RESPONSE] <your response>"
4. If there are no errors, say "[CORRECTION] No errors found. Perfect!" before responding
5. Keep your responses concise and natural'''}
    ]

if 'user_data' not in st.session_state:
    st.session_state.user_data = {
        'attempts': [], 'scores': [], 'correct_streak': 0,
        'last_correction': None, 'topic_history': deque(maxlen=5),
        'is_off_topic': False, 'last_relevant_topic': None,
        'consecutive_off_topic': 0
    }

if 'last_bot_question' not in st.session_state:
    st.session_state.last_bot_question = None

# === UTILITY FUNCTIONS ===
def extract_correction(bot_reply):
    match = re.search(r'\[CORRECTION\](.*?)(\n\n|$)', bot_reply, re.DOTALL)
    return None if not match or "no errors" in match.group(1).lower() else match.group(1).strip()

def analyze_grammar(original_text, bot_reply):
    correction = extract_correction(bot_reply)
    if correction is None:
        return 1.0
    o_words, c_words = original_text.lower().split(), correction.lower().split()
    diff_count = sum(1 for o, c in zip(o_words, c_words) if o != c) + abs(len(o_words) - len(c_words))
    return 1.0 - diff_count / max(len(o_words), len(c_words))

def predict_progress(scores):
    if len(scores) < 2:
        return None, None
    X = np.array(range(len(scores))).reshape(-1, 1)
    model = LinearRegression().fit(X, scores)
    return model.coef_[0], model.predict(np.array(range(len(scores), len(scores)+3)).reshape(-1, 1))

def update_sidebar():
    udata = st.session_state.user_data
    with st.sidebar:
        st.markdown("### 📊 User Progress Dashboard")
        if udata['scores']:
            st.metric("Current Score", f"{udata['scores'][-1]:.2f}", delta=f"{udata['scores'][-1] - (udata['scores'][-2] if len(udata['scores']) > 1 else 0):+.2f}")
            st.metric("Average Score", f"{np.mean(udata['scores']):.2f}")
            st.metric("Current Streak", udata['correct_streak'])

            fig, ax = plt.subplots()
            ax.plot(udata['attempts'], udata['scores'], 'o-', color='blue')
            trend, preds = predict_progress(udata['scores'])
            if preds is not None:
                ax.plot(range(len(udata['attempts']), len(udata['attempts'])+3), preds, 'r--')
            ax.set_ylim(0, 1.05)
            ax.set_xlabel("Attempt")
            ax.set_ylabel("Score")
            ax.grid(True)
            st.markdown("### Learning Progress")
            st.pyplot(fig)
        else:
            st.info("Start chatting to see progress!")

# === UI MESSAGE DISPLAY ===
for msg in st.session_state.messages[1:]:
    with st.chat_message(msg['role']):
        if '[CORRECTION]' in msg['content']:
            parts = msg['content'].split('[RESPONSE]')
            st.markdown(f"**Correction:** {parts[0].replace('[CORRECTION]', '').strip()}")
            if len(parts) > 1:
                st.markdown(f"**Response:** {parts[1].strip()}")
        else:
            st.markdown(msg['content'])

# === CHAT LOGIC ===
user_input = st.chat_input("Type your English sentence...")
if user_input:
    udata = st.session_state.user_data
    last_q = st.session_state.last_bot_question
    relevant, similarity = is_semantically_relevant(last_q, user_input) if last_q else (True, 1.0)

    st.session_state.messages.append({'role': 'user', 'content': user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.spinner("Analyzing..."):
        try:
            if not relevant:
                udata['is_off_topic'] = True
                udata['last_relevant_topic'] = last_q
                udata['consecutive_off_topic'] += 1

                st.warning(f"✖️ Your response is off-topic.\n\nLet's go back to: **{last_q}**")

                res = Generation.call(
                    model="qwen-plus",
                    messages=[
                        {"role": "system", "content": "Correct the English sentence. Only reply: [CORRECTION] <text>"},
                        {"role": "user", "content": user_input}
                    ],
                    result_format="message"
                )
                bot_reply = res.output.choices[0].message.content
                bot_reply = f"[CORRECTION] {bot_reply}" if "[CORRECTION]" not in bot_reply else bot_reply

                with st.chat_message("assistant"):
                    st.warning(bot_reply.replace("[CORRECTION]", "**Grammar correction:** "))
                st.session_state.messages.append({'role': 'assistant', 'content': bot_reply})
            else:
                udata['is_off_topic'] = False
                udata['consecutive_off_topic'] = 0

                res = Generation.call(
                    model="qwen-plus",
                    messages=st.session_state.messages,
                    result_format="message"
                )
                bot_reply = res.output.choices[0].message.content
                if '?' in bot_reply:
                    st.session_state.last_bot_question = bot_reply.split('[RESPONSE]')[-1].split('?')[0] + '?'

                with st.chat_message("assistant"):
                    if '[CORRECTION]' in bot_reply and '[RESPONSE]' in bot_reply:
                        corr, resp = bot_reply.split('[RESPONSE]')
                        st.markdown(f"**Correction:** {corr.replace('[CORRECTION]', '').strip()}")
                        st.markdown(f"**Response:** {resp.strip()}")
                    else:
                        st.markdown(bot_reply)

                score = analyze_grammar(user_input, bot_reply)
                udata['correct_streak'] = udata['correct_streak'] + 1 if score == 1.0 else 0
                final_score = min(score + (udata['correct_streak'] * 0.05), 1.0)
                udata['attempts'].append(len(udata['attempts']) + 1)
                udata['scores'].append(final_score)
                udata['last_correction'] = extract_correction(bot_reply)
                if score == 1.0:
                    st.success(f"✅ Perfect! Score: {final_score:.2f} | Streak: {udata['correct_streak']}")
                else:
                    st.info(f"Score: {final_score:.2f} | Streak reset to 0")
                st.session_state.messages.append({'role': 'assistant', 'content': bot_reply})
        except Exception as e:
            st.error(f"Something went wrong: {e}")

    update_sidebar()
