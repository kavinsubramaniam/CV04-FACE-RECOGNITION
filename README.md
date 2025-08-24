# CV04-FACE-RECOGNITION

# Attendance Manager using Face Recognition

A real-time attendance management system that uses **Face Recognition** for marking and tracking attendance. The system is designed for **accuracy and speed** using deep learning models, **ONNX optimization**, and **FAISS-based vector search**.

---

## 📌 Features
- **Face Capture via Flask UI** – Users can capture their face using a web interface.
- **Face Detection** – Implemented using **YOLOv11-FACE**, optimized with **ONNX** for faster inference.
- **Feature Extraction** – Utilizes **VGG-FACE (4096-D)** embeddings from **DEEPFACE** for accurate face representation.
- **Vector Database for Similarity Search** – Stores embeddings in **FAISS** for efficient nearest neighbor search.
- **Real-Time Attendance Marking** – Sub-second recognition latency for quick attendance updates.

---

## 🛠 Tech Stack
- **Frontend & Backend:** Flask (Python)
- **Face Detection:** YOLOv11-FACE (ONNX optimized)
- **Feature Extraction:** VGG-FACE (4096-dimensional embeddings) - **DEEPFACE**
- **Database:** FAISS (for vector similarity search)
- **Other Tools:** OpenCV, NumPy, ONNX Runtime

---

## ⚙️ System Workflow
1. **Capture Face** – The user’s face is captured via a webcam using a **Flask-based UI**.
2. **Face Detection** – YOLOv11-FACE detects the face in real-time. The YOLO model is converted to **ONNX format** for optimized inference.
3. **Feature Extraction** – Extract features using **VGG-FACE** model, resulting in a 4096-dimension embedding.
4. **Store in Vector Database** – Embeddings are stored in **FAISS index DB** for efficient similarity search.
5. **Attendance Marking** – During inference, the same process runs until feature extraction. The extracted vector is compared against stored vectors in FAISS to identify the user and mark attendance.

---

## 🔍 How It Works
### **Enrollment:**
Capture user face → Detect → Extract features → Store in FAISS index.

### **Attendance:**
Capture new face → Detect → Extract features → Compare with FAISS DB → Mark attendance if matched.

---

## 📦 Dependencies
- Flask
- OpenCV
- ONNX Runtime
- FAISS
- NumPy
- DeepFace

---

## 📈 Performance
- **Face Detection:** YOLOv11-FACE (ONNX optimized)
- **Feature Extraction:** VGG-FACE embeddings (4096-D)
- **Similarity Search:** FAISS index for real-time matching
- **Latency:** Sub-second recognition for stored users

---

## 🎯 Future Enhancements
- Implement **liveness detection** to prevent spoofing.
- Add **user management dashboard** for admin.

