from flask import Flask, render_template, Response, request, redirect, url_for, flash, jsonify
import cv2
import os
from deepface import DeepFace
import threading
import joblib

app = Flask(__name__)
app.secret_key = "supersecretkey"

dataset_path = "./static/registered_faces"
data_path = "student_data.pkl"
os.makedirs(dataset_path, exist_ok=True)

# -------------------------
# Load or Initialize Student Data
# -------------------------
if os.path.exists(data_path):
    student_data = joblib.load(data_path)
else:
    student_data = {}

# -------------------------
# Global State
# -------------------------
frame_lock = threading.Lock()
current_frame = None
recognized_name = "Detecting..."
recognized_info = {}

# -------------------------
# Face Recognition Thread
# -------------------------
def recognize_face():
    global current_frame, recognized_name, recognized_info
    while True:
        if current_frame is not None:
            with frame_lock:
                frame_copy = current_frame.copy()
                current_frame = None

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
                    
                    # Only update if a new name is found
                    recognized_name = name
                    recognized_info = student_data.get(name, {})
                else:
                    recognized_name = "Unknown"
                    recognized_info = {}
            except Exception as e:
                print("Recognition error:", e)
                recognized_name = "Unknown"
                recognized_info = {}
        else:
            # Optional: if needed, reset if no frame is ready for a while
            pass


# Start recognition thread
threading.Thread(target=recognize_face, daemon=True).start()

# -------------------------
# Video Streaming Generator
# -------------------------
camera = cv2.VideoCapture(0)

def gen_frames():
    global current_frame
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            with frame_lock:
                if current_frame is None:
                    current_frame = frame.copy()

            cv2.putText(frame, f"Name: {recognized_name}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# -------------------------
# Routes
# -------------------------
@app.route('/')
def index():
    return render_template('index.html', name=recognized_name, info=recognized_info)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_recognized_info')
def get_recognized_info():
    return jsonify({
        'name': recognized_name,
        'info': recognized_info
    })

@app.route('/register', methods=['POST'])
def register():
    name = request.form.get("name")
    roll = request.form.get("roll")
    dept = request.form.get("dept")

    if name and roll and dept:
        success, frame = camera.read()
        if success:
            person_dir = os.path.join(dataset_path, name)
            os.makedirs(person_dir, exist_ok=True)
            filename = os.path.join(person_dir, f"{name}_registered.jpg")
            cv2.imwrite(filename, frame)

            student_data[name] = {
                "roll": roll,
                "dept": dept
            }
            joblib.dump(student_data, data_path)

            flash(f"{name}'s face registered successfully.", "success")
        else:
            flash("Failed to capture image from webcam.", "danger")
    else:
        flash("All fields are required.", "warning")
    return redirect(url_for('index'))

# -------------------------
# Main
# -------------------------
if __name__ == '__main__':
    app.run(debug=True)
