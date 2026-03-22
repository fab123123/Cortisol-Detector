import streamlit as st
import cv2
import numpy as np
import requests
from CNN_Model.src.FaceCortisol import FaceCortisol
from datetime import datetime
import random

# ---------------------------------------------------------------------------
# Model import — swap this one line when your teammate's file is ready:
#   from cortisol_model import CortisolModel
from mock_model import MockCortisolModel as CortisolModel
# ---------------------------------------------------------------------------


@st.cache_resource
def load_model():
    """Load once, reuse across reruns."""
    return CortisolModel(model_path="model.h5")  # adjust path as needed


def get_contextual_data(city="Long Beach"):
    now = datetime.now()
    hour = now.hour
    #
    API_KEY = st.secrets.get("OWM_API_KEY", "your_free_api_key")
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
    #
    description = "Clear"
    temp = 20
    using_fallback = True
    return hour, temp, description, using_fallback
    #

def calculate_final_score(image_classification_prob, hour, temp, weather_desc):
    score = image_classification_prob

    if 6 <= hour <= 9:
        score -= 0.15

    if hour > 22 or hour < 4:
        score += 0.10

    if temp > 35 or weather_desc in ["Thunderstorm", "Rain", "Drizzle", "Snow"]:
        score += 0.05

    return clamp(score, 0.0, 1.0)


def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("🧠 MindCheck: Cortisol Detector")
st.write("Position your face in the center of the frame for the most accurate reading.")

model = FaceCortisol()

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
        face_roi = gray[y : y + h, x : x + w]
        resized_img = cv2.resize(face_roi, (48, 48), interpolation=cv2.INTER_AREA)

        with st.expander("Debug view"):
            st.image(resized_img, caption="Processed image (48x48)", width=150)

        st.success("Face detected and processed!")

        if st.button("Run Bio-Scan"):
            with st.spinner("Analyzing physiological markers..."):
                ai_prediction = model.predict(resized_img)
                ai_prediction = round(random.uniform(0,0.15),2) if ai_prediction=="low" else round(random.uniform(.7, 1))
                hour, temp, desc, using_fallback = get_contextual_data()

                if using_fallback:
                    st.caption("⚠️ Using fallback weather data (API unavailable).")

                final_val = calculate_final_score(ai_prediction, hour, temp, desc)
                st.session_state["last_result"] = final_val

            result = st.session_state["last_result"]
            if result > 0.7:
                st.warning(f"High Cortisol Detected ({int(result * 100)}%)")
                st.info("Tip: Try a 2-minute box breathing exercise.")
            else:
                st.success(f"Normal Levels Detected ({int(result * 100)}%)")
    else:
        st.error("No face detected. Please try again in better lighting!")