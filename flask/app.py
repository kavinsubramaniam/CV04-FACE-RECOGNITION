from flask import Flask, request, Response
import cv2
from flask_cors import CORS
import numpy as np
import threading

app = Flask(__name__)
CORS(app)

frame_lock = threading.Lock()
latest_frame = None

@app.route('/stream', methods=['POST'])
def receive_frame():
    global latest_frame

    if 'frame' not in request.files:
        print("No frame received!")
        return "No frame received", 400

    file = request.files['frame']
    print(f"Received frame: {file.filename}, Size: {len(file.read())} bytes")  # Debugging log
    file.seek(0)  # Reset file pointer

    nparr = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        print("Error decoding frame!")
        return "Error decoding frame", 400

    with frame_lock:
        latest_frame = frame

    return "Frame received", 200

def generate_frames():
    while True:
        with frame_lock:
            if latest_frame is None:
                continue
            _, buffer = cv2.imencode('.jpg', latest_frame)
            frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)
