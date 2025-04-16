import logging
import av
import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import os
import pickle

# Setup Logging
logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)

# Load Face Detection and Recognition Pipelines
from src.face_detection import FaceDetection
from pipelines import FaceRecognitionPipeline

face_recognition_pipeline = FaceRecognitionPipeline()
model_path = "models/yolov11n-face.onnx"
face_detection = FaceDetection(model_path)

# Load or Initialize User Mappings
user_mappings_path = "face_embeddings/user_mappings.pkl"
if os.path.exists(user_mappings_path):
    with open(user_mappings_path, "rb") as f:
        user_mappings = pickle.load(f)
else:
    user_mappings = {}

# Streamlit UI
st.title("Face Recognition App")
st.sidebar.header("Options")

mode = st.sidebar.radio("Choose Mode", ["Register", "Recognize"])
user_name = st.sidebar.text_input("Enter your Name") if mode == "Register" else None


class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")  # Convert AV frame to NumPy array
        logging.info("Received a frame for processing")  # ✅ Log frame reception

        detected_face = face_detection.detect_faces(img)
        
        if detected_face:
            logging.info("Face detected!")  # ✅ Log face detection
            x, y, w, h, _, _ = detected_face
            face_img = img[y:y+h, x:x+w]  # Crop face
            
            if mode == "Register":
                if user_name:
                    face_recognition_pipeline.register(face_img)
                    user_mappings[user_name] = len(user_mappings)
                    label = f"Registered: {user_name}"
                else:
                    label = "Enter Name to Register"
            else:
                label = face_recognition_pipeline.recognize(face_img)

            # Ensure label is a string and face_img is a NumPy array
            if not isinstance(face_img, np.ndarray):
                raise TypeError(f"face_img is not a NumPy array. Found type: {type(face_img)}")

            # Draw Rectangle and Label
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            logging.info("No face detected in the frame")  # ✅ Log missing detection
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")



# WebRTC Video Stream
# webrtc_streamer(key="face-recognition", video_transformer_factory=VideoTransformer)
webrtc_streamer(
    key="face-recognition",
    # video_frame_callback=None,
    video_transformer_factory=VideoTransformer,
    rtc_configuration={
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},  # Google's public STUN server
            {"urls": ["turn:turn.server.com"], "username": "user", "credential": "pass"}  # Use TURN if needed
        ]
    }
)
# Save user mappings
with open(user_mappings_path, "wb") as f:
    pickle.dump(user_mappings, f)
