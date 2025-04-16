import streamlit as st
import cv2
import threading
import os
from deepface import DeepFace
from PIL import Image
import numpy as np

# -------------------------
# Configuration
# -------------------------
dataset_path = "./test/database"
frame_for_recognition = None
recognized_name = "Detecting..."
lock = threading.Lock()

# Ensure dataset path exists
os.makedirs(dataset_path, exist_ok=True)

# -------------------------
# Face Recognition Thread
# -------------------------
def recognize_face():
    global frame_for_recognition, recognized_name
    while True:
        if frame_for_recognition is not None:
            with lock:
                frame_copy = frame_for_recognition.copy()
                frame_for_recognition = None

            try:
                result = DeepFace.find(
                    img_path=cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB),
                    db_path=dataset_path,
                    silent=True,
                    enforce_detection=False,
                    refresh_database=True
                )

                if result and not result[0].empty:
                    identity_path = result[0]['identity'].iloc[0]
                    name = os.path.basename(os.path.dirname(identity_path))
                    recognized_name = name
                    st.session_state.recognized_name = name
                else:
                    recognized_name = "Unknown"
                    st.session_state.recognized_name = "Unknown"
            except Exception as e:
                recognized_name = "Error"
                st.session_state.recognized_name = "Error"
                print(f"Recognition Error: {e}")
            print(recognized_name)

# -------------------------
# Initialize Session States
# -------------------------
if 'recognition_thread_started' not in st.session_state:
    threading.Thread(target=recognize_face, daemon=True).start()
    st.session_state.recognition_thread_started = True

if 'recognized_name' not in st.session_state:
    st.session_state.recognized_name = "Detecting..."

if 'stop_stream' not in st.session_state:
    st.session_state.stop_stream = False

# -------------------------
# Streamlit UI
# -------------------------
st.title("🎥 Real-Time Face Recognition")

image_placeholder = st.empty()
name_placeholder = st.empty()

# -------------------------
# Sidebar for Registration
# -------------------------
st.sidebar.header("📝 Face Registration")
new_name = st.sidebar.text_input("Enter Name to Register")
register_button = st.sidebar.button("📸 Register Face", key="register_face")

# Stop button
if st.button("🛑 Stop Stream", key="stop_btn"):
    st.session_state.stop_stream = True

# -------------------------
# Start Webcam
# -------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    st.error("Webcam not detected.")
else:
    while cap.isOpened() and not st.session_state.stop_stream:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture image from webcam.")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame_rgb)

        with lock:
            if frame_for_recognition is None:
                frame_for_recognition = frame.copy()

        image_placeholder.image(img_pil, channels="RGB")
        name_placeholder.markdown(f"**Recognized Name:** `{st.session_state.recognized_name}`")

        # Register the face if button clicked
        if register_button and new_name.strip():
            person_dir = os.path.join(dataset_path, new_name.strip())
            os.makedirs(person_dir, exist_ok=True)
            filename = os.path.join(person_dir, f"{new_name.strip()}_registered.jpg")
            cv2.imwrite(filename, frame)
            st.sidebar.success(f"{new_name.strip()}'s face registered.")
            st.rerun()

# -------------------------
# Release Resources
# -------------------------
cap.release()
cv2.destroyAllWindows()
