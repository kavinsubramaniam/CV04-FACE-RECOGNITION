from deepface import DeepFace
import cv2
import threading

# Global variable to store the latest frame
latest_frame = None
lock = threading.Lock()

def face_det_emd():
    """ Runs in a separate thread to process face embeddings. """
    global latest_frame
    while True:
        with lock:
            if latest_frame is None:
                continue  # Skip if there's no new frame

            frame_rgb = cv2.cvtColor(latest_frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            latest_frame = None  # Clear to avoid duplicate processing

        try:
            face_embedding = DeepFace.represent(
                img_path=frame_rgb,  # Pass the frame as an array
                model_name="Facenet512",
                enforce_detection=False,
                detector_backend="retinaface",
                align=True,
                max_faces=1
            )
            print(face_embedding)

        except Exception as e:
            print(f"Error in face detection: {e}")

def video_capture():
    """ Captures video frames and sends them to the face detection thread. """
    global latest_frame
    cap = cv2.VideoCapture(0)

    # Start the face detection thread
    thread = threading.Thread(target=face_det_emd, daemon=True)
    thread.start()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frames.")
            break

        with lock:
            latest_frame = frame.copy()  # Store the latest frame safely

        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    video_capture()

if __name__ == "__main__":
    main()
