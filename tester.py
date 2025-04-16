# import logging
# import av
# import cv2
# import numpy as np
# import gradio as gr
# import os
# import pickle

# # Setup Logging
# logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)

# # Load Face Detection and Recognition Pipelines
# from src.face_detection import FaceDetection
# from pipelines import FaceRecognitionPipeline

# face_recognition_pipeline = FaceRecognitionPipeline()
# model_path = "models/yolov11n-face.onnx"
# face_detection = FaceDetection(model_path)

# # Load or Initialize User Mappings
# user_mappings_path = "face_embeddings/user_mappings.pkl"
# if os.path.exists(user_mappings_path):
#     with open(user_mappings_path, "rb") as f:
#         user_mappings = pickle.load(f)
# else:
#     user_mappings = {}

# logging.info(f"User Mappping {user_mappings}")

# # Face Recognition Function
# def process_frame(frame, mode, user_name=""):
#     # img = np.array(frame, dtype=np.uint8)  # Convert Gradio frame to NumPy array
#     if not frame: return frame
#     cap = cv2.VideoCapture(frame)  # Open the video file
#     ret, img = cap.read()  # Read the first frame
#     cap.release()
    
#     logging.info("Received a frame for processing")  # ✅ Log frame reception

#     detected_face = face_detection.detect_faces(img)

#     if detected_face:
#         logging.info(f"Face detected! {detected_face}")  # ✅ Log face detection
#         x, y, w, h, _, _ = detected_face
#         face_img = img[y:y+h, x:x+w]  # Crop face

#         if mode == "Register":
#             if user_name:
#                 face_recognition_pipeline.register(face_img)
#                 user_mappings[len(user_mappings)] = user_name
#                 label = f"Registered: {user_name}"
#             else:
#                 label = "Enter Name to Register"
#         else:
#             label = face_recognition_pipeline.recognize(face_img)
#             logging.info(f"{label}")
#             label = user_mappings[int(label)]

#         # Draw Rectangle and Label
#         cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
#         cv2.putText(img, str(label), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
#     else:
#         logging.info("No face detected in the frame")  # ✅ Log missing detection

#     return img

# # Gradio UI
# def face_recognition_app(mode, user_name, video):
#     return process_frame(video, mode, user_name)

# iface = gr.Interface(
#     fn=face_recognition_app,
#     inputs=[
#         gr.Radio(["Register", "Recognize"], label="Mode"),
#         gr.Textbox(label="Enter Name (Only for Registration)"),
#         gr.Video()  # ✅ Removed `source="webcam"` argument
#     ],
#     outputs=gr.Image(),
#     live=True
# )

# # Save user mappings on exit
# with open(user_mappings_path, "wb") as f:
#     pickle.dump(user_mappings, f)

# # Launch Gradio App
# iface.launch()


import gradio as gr
import cv2
import numpy as np

def generate_frames():
    """Captures video from webcam and applies processing in real-time."""
    cap = cv2.VideoCapture(0)  # Open webcam
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Example: Add overlay (Replace this with face recognition logic)
        cv2.putText(frame, "Live Processing", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        _, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()

        yield frame_bytes  # Yield each processed frame as bytes

    cap.release()

iface = gr.Interface(
    fn=generate_frames,
    inputs=gr.Video(),
    outputs=gr.Video(format="mp4", streaming=True),  # Video output stream
    live=True
)

iface.launch()
