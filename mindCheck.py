import streamlit as st
import streamlit.components.v1 as components
import cv2
import numpy as np
import requests
from datetime import datetime
import csv
import random

from CNN_Model.src.FaceCortisol import FaceCortisol

ASI1_API_KEY = "sk_016e899f8efc4503bdff5a1d0e221b548bd8c281827e420d9c755523c3b8291e"

SYSTEM_PROMPT = """You are MindCheck, a warm and knowledgeable AI stress and cortisol wellness assistant.

You help users understand and manage their stress levels based on their facial scan results.

Your capabilities:
- Interpret cortisol/stress scan percentages
- Suggest breathing exercises, grounding techniques, and lifestyle tips
- Answer follow-up questions about stress, cortisol, sleep, anxiety, and wellness
- Provide personalized advice based on weather and time context

Rules:
- Be conversational, warm, and supportive — not clinical
- Keep responses concise (2-4 sentences unless user asks for more detail)
- Remember the user's scan result and weather context throughout the conversation
- Never make medical diagnoses — always suggest professional help for serious concerns
"""

def call_asi1(messages: list) -> str | None:
    """Call ASI:One with full conversation history for multi-turn chat."""
    try:
        response = requests.post(
            "https://api.asi1.ai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {ASI1_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "asi1-mini",
                "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + messages,
                "max_tokens": 300,
                "temperature": 0.7
            },
            timeout=20
        )
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            print(f"[ASI1 error] {response.status_code}: {response.text}")
            return None
    except Exception as e:
        print(f"[ASI1 exception] {e}")
        return None


# ---------------------------------------------------------------------------
# Music URLs
# ---------------------------------------------------------------------------
MUSIC = {
    "high":   "https://www.youtube.com/embed/xZfV8SP0Pto?autoplay=1",
    "medium": "https://www.youtube.com/embed/CLgkjTVyqHo?autoplay=1",
    "low":    "https://www.youtube.com/embed/NOd291dK1Do?autoplay=1",
}

# ---------------------------------------------------------------------------
# Gauge
# ---------------------------------------------------------------------------
def render_gauge(score_pct: int):
    arc_len = 346
    offset = arc_len * (1 - score_pct / 100)
    angle = -90 + (score_pct / 100 * 180)

    if score_pct > 70:
        status_text = "high — try box breathing"
        status_color = "#E24B4A"
    elif score_pct > 40:
        status_text = "moderate"
        status_color = "#EF9F27"
    else:
        status_text = "low"
        status_color = "#1D9E75"

    components.html(f"""
    <div style="display:flex;flex-direction:column;align-items:center;padding:1rem 0;">
      <svg viewBox="0 0 260 150" width="260" height="150" xmlns="http://www.w3.org/2000/svg">
        <defs>
          <linearGradient id="arcGrad" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%"   stop-color="#1D9E75"/>
            <stop offset="50%"  stop-color="#EF9F27"/>
            <stop offset="100%" stop-color="#E24B4A"/>
          </linearGradient>
        </defs>
        <path d="M 20 130 A 110 110 0 0 1 240 130"
              fill="none" stroke="#e0e0e0" stroke-width="18" stroke-linecap="round"/>
        <path d="M 20 130 A 110 110 0 0 1 240 130"
              fill="none" stroke="url(#arcGrad)" stroke-width="18" stroke-linecap="round"
              stroke-dasharray="{arc_len}" stroke-dashoffset="{offset:.1f}"/>
        <line x1="130" y1="130" x2="130" y2="38"
              stroke="#333" stroke-width="2.5" stroke-linecap="round"
              transform="rotate({angle:.1f} 130 130)"/>
        <circle cx="130" cy="130" r="6" fill="#333"/>
        <text x="18"  y="148" font-size="11" fill="#888">0</text>
        <text x="118" y="22"  font-size="11" fill="#888">50</text>
        <text x="234" y="148" font-size="11" fill="#888">100</text>
      </svg>
      <div style="font-size:36px;font-weight:500;margin-top:0.5rem;">{score_pct}%</div>
      <div style="font-size:14px;font-weight:500;color:{status_color};margin-top:0.25rem;">{status_text}</div>
    </div>
    """, height=220)


# ---------------------------------------------------------------------------
# ML Model
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    return FaceCortisol()


# ---------------------------------------------------------------------------
# Weather
# ---------------------------------------------------------------------------
def get_contextual_data(city="Long Beach"):
    now_hour = datetime.now().hour
    try:
        API_KEY = st.secrets.get("OWM_API_KEY", "your_free_api_key")
    except Exception:
        API_KEY = "your_free_api_key"

    url = (
        f"http://api.openweathermap.org/data/2.5/weather"
        f"?q={city}&appid={API_KEY}&units=metric"
    )
    try:
        response = requests.get(url, timeout=5).json()
        temp = response["main"]["temp"]
        description = response["weather"][0]["main"]
        using_fallback = False
    except (requests.RequestException, KeyError) as e:
        print(f"[Weather API error] {e}")
        temp, description = 20, "Clear"
        using_fallback = True

    return now_hour, temp, description, using_fallback


# ---------------------------------------------------------------------------
# Score
# ---------------------------------------------------------------------------
def calculate_final_score(image_classification_prob, hour, temp, weather_desc):
    score = image_classification_prob

    if 6 <= hour <= 9:
        score -= 0.15
    elif hour > 22 or hour < 4:
        score += 0.10

    if temp > 35 or weather_desc in ["Thunderstorm", "Rain", "Drizzle", "Snow"]:
        score += 0.05
    else:
        score -= 0.05

    return max(min(1.0, score), 0.0)


# ---------------------------------------------------------------------------
# Image helper
# ---------------------------------------------------------------------------
def load_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    st.image(img)


# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------
if "started" not in st.session_state:
    st.session_state["started"] = False
if "scan_done" not in st.session_state:
    st.session_state["scan_done"] = False
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "scan_context" not in st.session_state:
    st.session_state["scan_context"] = ""


# ---------------------------------------------------------------------------
# Welcome screen
# ---------------------------------------------------------------------------
if not st.session_state["started"]:
    st.title("🧠 MindCheck: Cortisol Detector")
    st.subheader("Your AI-powered stress level analyzer")
    st.write("""
        MindCheck uses your facial expression and environmental data
        to estimate your current cortisol (stress) levels.
    """)
    st.info("""
        **How it works:**
        1. Allow camera access when prompted
        2. Position your face in the center of the frame
        3. Take a snapshot and run the Bio-Scan
        4. Get your personalized stress report + chat with the AI
    """)
    st.warning("📷 Camera access is required. No images are stored or sent to any server.")
    if st.button("Get Started →"):
        st.session_state["started"] = True
        st.rerun()

# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------
else:
    st.title("🧠 MindCheck: Cortisol Detector")

    if st.button("← Back"):
        st.session_state["started"] = False
        st.session_state["scan_done"] = False
        st.session_state["chat_history"] = []
        st.session_state["scan_context"] = ""
        st.rerun()


    # PHASE 1 — SCAN (only shown before scan is complete)
    if not st.session_state["scan_done"]:
        st.write("Position your face in the center of the frame for the most accurate reading.")
        model = load_model()
        img_file = st.camera_input("Take a health snapshot")

        if img_file:
            bytes_data = img_file.getvalue()
            cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            if len(faces) > 0:
                (x, y, w, h) = faces[0]
                face_roi = gray[y: y + h, x: x + w]
                resized_img = cv2.resize(face_roi, (48, 48), interpolation=cv2.INTER_AREA)

                with st.expander("Debug view"):
                    st.image(resized_img, caption="Processed image (48x48)", width=150)

                st.success("Face detected and processed!")

                if st.button("Run Bio-Scan"):
                    with st.spinner("Analyzing physiological markers..."):
                        ai_prediction = model.predict(resized_img)
                        hour, temp, desc, using_fallback = get_contextual_data()

                        if using_fallback:
                            st.caption("⚠️ Using fallback weather data.")

                        print(f"[Debug] hour={hour}, temp={temp}, desc={desc}, fallback={using_fallback}")
                        print(f"[Debug] ai_prediction={ai_prediction:.2f}")

                        final_val = calculate_final_score(ai_prediction, hour, temp, desc)
                        score_pct = int(final_val * 100)

                        st.session_state["last_result"] = final_val
                        st.session_state["score_pct"] = score_pct
                        st.session_state["weather_desc"] = desc
                        st.session_state["weather_temp"] = temp

                        # Build scan context and get initial AI response
                        scan_context = (
                            f"I just completed my MindCheck facial stress scan. "
                            f"My detected stress/cortisol level is {score_pct}%. "
                            f"Current weather: {desc}, {temp}°C. "
                            f"Please give me an initial assessment and your first tip."
                        )
                        st.session_state["scan_context"] = scan_context

                        initial_reply = call_asi1([{"role": "user", "content": scan_context}])

                        if initial_reply:
                            st.session_state["chat_history"] = [
                                {"role": "user", "content": scan_context},
                                {"role": "assistant", "content": initial_reply}
                            ]

                        st.session_state["scan_done"] = True
                        st.rerun()
            else:
                st.error("No face detected. Please try again in better lighting!")

    # PHASE 2 — RESULTS + PERSISTENT CHAT (shown after scan is complete)
    else:
        result    = st.session_state["last_result"]
        score_pct = st.session_state["score_pct"]
        desc      = st.session_state["weather_desc"]
        temp      = st.session_state["weather_temp"]

        # Gauge
        render_gauge(score_pct)

        # Result banner
        if result > 0.7:
            load_image("assets/images/high_cortisol.jpg")
            USER_IMAGE = "assets/images/high_cortisol.jpg"
            st.warning(f"High Cortisol Detected ({score_pct}%)")
        elif result > 0.4:
            load_image("assets/images/medium_cortisol.jpg")
            USER_IMAGE = "assets/images/medium_cortisol.jpg"
            st.info(f"Moderate Cortisol Detected ({score_pct}%)")
        else:
            load_image("assets/images/low_cortisol.jpg")
            USER_IMAGE = "assets/images/low_cortisol.jpg"
            st.success(f"Low Levels Detected ({score_pct}%)")

        # Random CSV advice
        with open("advice.csv", newline='', encoding='utf-8') as f:
            advice_rows = list(csv.reader(f))
        st.info(random.choice(advice_rows)[0])

        # Music player
        music_key = "high" if result > 0.7 else "medium" if result > 0.4 else "low"
        components.html(f"""
            <iframe width="640" height="100" src="{MUSIC[music_key]}"
            frameborder="0" allow="autoplay; encrypted-media"></iframe>
        """, height=120)

        st.divider()


        # Chatbot "MindCheck"
        st.subheader("🤖 Chat with MindCheck AI")
        st.caption("Ask follow-up questions about your stress, breathing exercises, sleep tips, and more.")

        # Here we render full chat history
        for msg in st.session_state["chat_history"]:
            if msg["role"] == "user" and msg["content"] == st.session_state["scan_context"]:
                with st.chat_message("user", avatar = USER_IMAGE):
                    st.write(f"📊 Scan complete — stress level: **{score_pct}%**, weather: {desc}, {temp}°C")
            else:
                with st.chat_message(msg["role"], avatar= "assets/images/MindCheckAvatar.jpg"):
                    st.write(msg["content"])

        # Input box
        user_input = st.chat_input("Ask anything about your stress, wellness, breathing tips...")

        if user_input:
            with st.chat_message("user", avatar = USER_IMAGE):
                st.write(user_input)

            st.session_state["chat_history"].append({
                "role": "user",
                "content": user_input
            })

            with st.chat_message("assistant", avatar = "assets/images/MindCheckAvatar.jpg"):
                with st.spinner("Thinking..."):
                    reply = call_asi1(st.session_state["chat_history"])

                if reply:
                    st.write(reply)
                    st.session_state["chat_history"].append({
                        "role": "assistant",
                        "content": reply
                    })
                else:
                    st.error("Could not reach AI. Check your ASI1_API_KEY.")

        # Button for user to make a new scan
        st.divider()
        if st.button("🔄 New Scan"):
            st.session_state["scan_done"] = False
            st.session_state["chat_history"] = []
            st.session_state["scan_context"] = ""
            st.rerun()