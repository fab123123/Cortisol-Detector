import streamlit as st
import streamlit.components.v1 as components
import cv2
import numpy as np
import requests
from CNN_Model.src.FaceCortisol import FaceCortisol
from datetime import datetime
import random
import csv

# ---------------------------------------------------------------------------
# Music URLs — replace video IDs with your chosen tracks per stress level
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
# Model
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

    if 6 <= hour <= 9: # good hours lower cortisol
        score -= 0.15
    elif hour > 22 or hour < 4: # Late hours raise cortisol
        score += 0.10

    if temp > 35 or weather_desc in ["Thunderstorm", "Rain", "Drizzle", "Snow"]: # Bad weather raises cortisol
        score += 0.05
    else:
        score -= 0.05

    return clamp(score, 0.0, 1.0)


def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)


# ---------------------------------------------------------------------------
# Welcome screen
# ---------------------------------------------------------------------------
if "started" not in st.session_state:
    st.session_state["started"] = False

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
        4. Get your personalized stress report
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
    st.write("Position your face in the center of the frame for the most accurate reading.")

    if st.button("← Back"):
        st.session_state["started"] = False
        st.rerun()

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
                    ai_prediction = (
                        round(random.uniform(0, 0.15), 2)
                        if ai_prediction == "low"
                        else round(random.uniform(0.7, 1.0), 2)
                    )

                    hour, temp, desc, using_fallback = get_contextual_data()

                    if using_fallback:
                        st.caption("⚠️ Using fallback weather data (API unavailable).")

                    print(f"[Debug] hour={hour}, temp={temp}, desc={desc}, fallback={using_fallback}")
                    print(f"[Debug] ai_prediction={ai_prediction:.2f}, final_score={calculate_final_score(ai_prediction, hour, temp, desc):.2f}")

                    final_val = calculate_final_score(ai_prediction, hour, temp, desc)
                    st.session_state["last_result"] = final_val

                result = st.session_state["last_result"]
                score_pct = int(result * 100)

                # Gauge
                render_gauge(score_pct)

                # Result + music
                # Read CSV into a list
                with open("advice.csv", newline='', encoding='utf-8') as f:
                    advice_reader = csv.reader(f)
                    advice_rows = list(advice_reader)
                random_advice = random.choice(advice_rows)[0]

                if result > 0.7:
                    st.warning(f"High Cortisol Detected ({score_pct}%)")
                    st.info(random_advice)
                    components.html(f"""
                        <iframe width="300" height="80" src="{MUSIC['high']}"
                        frameborder="0" allow="autoplay; encrypted-media"></iframe>
                    """, height=90)
                elif result > 0.4:
                    st.info(f"Moderate Cortisol Detected ({score_pct}%)")
                    st.info(random_advice)
                    components.html(f"""
                        <iframe width="300" height="80" src="{MUSIC['medium']}"
                        frameborder="0" allow="autoplay; encrypted-media"></iframe>
                    """, height=90)
                else:
                    st.success(f"Low Levels Detected ({score_pct}%)")
                    components.html(f"""
                        <iframe width="300" height="80" src="{MUSIC['low']}"
                        frameborder="0" allow="autoplay; encrypted-media"></iframe>
                    """, height=90)
        else:
            st.error("No face detected. Please try again in better lighting!")