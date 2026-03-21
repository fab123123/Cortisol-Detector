import streamlit as st
import cv2
import numpy as np
#from PIL import Image
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

        # 4. Send to Classifier (Placeholder for your ML function)
        # prediction = my_classifier.predict(resized_img)
        # st.metric("Estimated Stress Level", f"{prediction}%")
    else:
        st.error("No face detected. Please try again in better lighting!")