# from deepface import DeepFace
# import cv2
# import os


# dataset_path = "../test/database"
# cap = cv2.VideoCapture(0)
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("Failed to capture frames.")
#         break

#     cv2.imshow("Face Recognition", frame)
#     result = DeepFace.find(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), db_path=dataset_path, silent=True,enforce_detection=False)
#     if result and not result[0].empty:
#         # Extract the first matched identity path
#         identity_path = result[0]['identity'].iloc[0]
#         # Get the name (folder name)
#         name = os.path.basename(os.path.dirname(identity_path))
#         print(name)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break


# DeepFace.stream(
#     db_path = dataset_path, 
#     enable_face_analysis = False,
#     frame_threshold = 0,
#     time_threshold = 0
#     )



# from deepface import DeepFace
# import cv2
# import os
# import threading

# dataset_path = "../test/database"
# cap = cv2.VideoCapture(0)

# recognized_name = None
# frame_for_recognition = None
# lock = threading.Lock()

# def recognize_face():
#     global frame_for_recognition, recognized_name
#     while True:
#         if frame_for_recognition is not None:
#             with lock:
#                 frame_copy = frame_for_recognition.copy()
#                 frame_for_recognition = None  # Reset so we only process once

#             result = DeepFace.find(
#                 img_path=cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB),
#                 db_path=dataset_path,
#                 silent=True,
#                 enforce_detection=False
#             )

#             if result and not result[0].empty:
#                 identity_path = result[0]['identity'].iloc[0]
#                 name = os.path.basename(os.path.dirname(identity_path))
#                 recognized_name = name
#             else:
#                 recognized_name = None

# # Start the recognition thread
# thread = threading.Thread(target=recognize_face, daemon=True)
# thread.start()

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("Failed to capture frames.")
#         break

#     # Send a frame for recognition every few seconds or frames
#     with lock:
#         if frame_for_recognition is None:
#             frame_for_recognition = frame.copy()

#     # Optionally draw the name on the frame
#     if recognized_name:
#         cv2.putText(frame, recognized_name, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#     cv2.imshow("Face Recognition", frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()


# ________________________________________________________
import streamlit as st
import cv2
import threading
import os
from deepface import DeepFace
from PIL import Image
import numpy as np

dataset_path = "../test/database"

frame_for_recognition = None
recognized_name = "Detecting..."
lock = threading.Lock()

# Recognition thread function
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
                else:
                    recognized_name = "Unknown"
            except Exception:
                recognized_name = "Error"

# Start recognition thread once
if 'recognition_thread_started' not in st.session_state:
    threading.Thread(target=recognize_face, daemon=True).start()
    st.session_state.recognition_thread_started = True

# Stop flag
if 'stop_stream' not in st.session_state:
    st.session_state.stop_stream = False

st.title("🎥 Real-Time Face Recognition")

image_placeholder = st.empty()
name_placeholder = st.empty()

# -------------------------
# Sidebar for Registration
# -------------------------
st.sidebar.header("📝 Face Registration")
new_name = st.sidebar.text_input("Enter Name to Register")
register_button = st.sidebar.button("📸 Register Face", key="register_face")


# Stop button — outside the loop
if st.button("Stop Stream", key="stop_btn"):
    st.session_state.stop_stream = True

# Start webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    st.error("❌ Unable to access the camera. Please check your webcam or close other apps using it.")
    raise "Camera is not accessable!"

# Loop until stop is pressed
while cap.isOpened() and not st.session_state.stop_stream:
    ret, frame = cap.read()
    if not ret:
        st.error("Failed to access webcam.")
        break

    frame_display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(frame_display)

    with lock:
        if frame_for_recognition is None:
            frame_for_recognition = frame.copy()

    image_placeholder.image(img_pil, channels="RGB")
    name_placeholder.markdown(f"**Recognized Name:** {recognized_name}")

     # Register the face if button clicked
    if register_button and new_name.strip():
        person_dir = os.path.join(dataset_path, new_name.strip())
        os.makedirs(person_dir, exist_ok=True)
        filename = os.path.join(person_dir, f"{new_name.strip()}_registered.jpg")
        cv2.imwrite(filename, frame)
        st.sidebar.success(f"{new_name.strip()}'s face registered.")
        st.rerun()

# Release resources
cap.release()
cv2.destroyAllWindows()
