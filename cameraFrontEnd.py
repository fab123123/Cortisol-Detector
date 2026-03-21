import streamlit as st
import cv2
import numpy as np
#from PIL import Image
import requests
from datetime import datetime

def get_contextual_data(city="Long Beach"):
    # Get Time info
    now = datetime.now()
    hour = now.hour  # 0-23

    # Replace 'YOUR_API_KEY' with a real key or hardcode for the demo
    API_KEY = "your_free_api_key"
    # API_KEY = "e4cd84d41d01e0d10a7182f4d883d941" Omar's API Key
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"

    try:
        response = requests.get(url).json()
        temp = response['main']['temp']
        description = response['weather'][0]['main']  # e.g., 'Rain', 'Clouds'
    except:
        temp, description = 20, "Clear"  # Fallback defaults

    return hour, temp, description


def calculate_final_score(image_classification_prob, hour, temp, weather_desc):
    # Start with the AI's confidence (0.0 to 1.0)
    score = image_classification_prob

    # Logic: If it's 8 AM, a 'high' reading is actually normal (reduce score)
    if 6 <= hour <= 9:
        score -= 0.15

        # Logic: If it's midnight and the AI sees stress, it's more concerning
    if hour > 22 or hour < 4:
        score += 0.10

    # Logic: Environmental stressors
    if temp > 35 or weather_desc in ["Thunderstorm", "Rain"]:
        score += 0.05

    return clamp(score, 0, 1)  # Keep it between 0 and 1


def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)

# Uses streamlit to allow web-application services to be quick and easy
st.title("🧠 MindCheck: Cortisol Detector")
st.write("Position your face in the center of the frame for the most accurate reading.")

# 1. Camera GUI
img_file = st.camera_input("Take a health snapshot")

if img_file:
    # Convert Streamlit file to OpenCV image
    bytes_data = img_file.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    # 2. Focus on Face (Haar Cascade)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    if len(faces) > 0:
        # Get the first face found (x, y, width, height)
        (x, y, w, h) = faces[0]
        face_roi = gray[y:y + h, x:x + w]

        # 3. Transform to Grayscale 48x48
        resized_img = cv2.resize(face_roi, (48, 48), interpolation=cv2.INTER_AREA)

        st.image(resized_img, caption="Processed Image (48x48)", width=150)
        st.success("Face detected and processed!")

        if st.button("Run Bio-Scan"):
            with st.spinner("Analyzing physiological markers..."):
                # 5. Send image to your teammate's model
                # ai_prediction = teammate_model.predict(resized_img)
                ai_prediction = 0.65  # Dummy value for now
                # 6. Take weather and time
                h, t, desc = get_contextual_data()

                # 7 & 8. Calculate Final Score
                final_val = calculate_final_score(ai_prediction, h, t, desc)

                # 9. Return information
                if final_val > 0.7:
                    st.warning(f"High Cortisol Detected ({int(final_val * 100)}%)")
                    st.info("Tip: Try a 2-minute box breathing exercise.")
                else:
                    st.success(f"Normal Levels Detected ({int(final_val * 100)}%)")
    else:
        st.error("No face detected. Please try again in better lighting!")
