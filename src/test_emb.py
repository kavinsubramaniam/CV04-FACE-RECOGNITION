import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pickle
import faiss
from face_embedding import FaceEmbeddings
import cv2


index = FaceEmbeddings("./")
index.create_index(shape=512)


model_name = ["Facenet", "Facenet512","OpenFace", "DeepFace", "DeepID", "Dlib", "ArcFace", "SFace","GhostFaceNet"]
model_name_f = model_name[1]

# Load your embeddings (shape: num_samples x 4096)
with open(f"face_embedding_{model_name_f}.pkl", "rb") as file:
    data = pickle.load(file)

with open("label.pkl", "rb") as file:
    label = pickle.load(file)

arvind_count = label.count("ArvindVikram")
kavin_count = label.count("Kavin")
kavin_sub_count = label.count("KavinSubramaniam")
rash_count = label.count("Rashwanth")



embeddings = [i[0]['embedding'] for i in data]
print(len(embeddings[0]))
print(f"The length of the embeddings : {len(embeddings)}")
print(f"The length of the labels : {len(label)}")




embeddings = np.array(embeddings)
index.insert(embeddings[:arvind_count], name="ArvindVikram")
index.insert(embeddings[:kavin_count], name="Kavin")

# index.write_index()
print("Count of arvind", arvind_count)
print(len(embeddings[0]))
# embedding = np.array(embedding).reshape(1, -1)
print(index.search_embedding(embeddings[1200].reshape(1, -1), k=10))




# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
from face_detection import FaceDetection
detection_model_path = "../models/yolov11n-face.onnx"
face_detection = FaceDetection(detection_model_path)

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read() 

    if not ret:
        print("Failed to capture frame.")
        break

    # Detect faces
    face = face_detection.detect_faces(frame)

    if face:
        x1, y1, x2, y2, _, _ = face
        face_img = frame[y1:y2, x1:x2]
        name = index.search_face(frame, k=10, backend_model="Facenet512", threshold=1.55)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        print(f"{name}")
        
    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++







# label_color = ["red", "yellow", "green", "black"]
# label_color_mappings = []

# for i in label:
#     if i == "Kavin": label_color_mappings.append(label_color[0])
#     elif i == "Rashwanth": label_color_mappings.append(label_color[1])
#     elif i == "KavinSubramaniam": label_color_mappings.append(label_color[2])
#     else: label_color_mappings.append(label_color[3])


# embeddings = np.array(embeddings)

# mean_vector = np.mean(embeddings, axis=0)
# variance_vector = np.var(embeddings, axis=0)

# print("Mean of embeddings:", np.mean(mean_vector))
# print("Variance of embeddings:", np.mean(variance_vector))

# plt.hist(embeddings.flatten(), bins=50)
# plt.title("Distribution of Embedding Values")
# plt.xlabel("Embedding Value")
# plt.ylabel("Frequency")
# plt.show()

# from sklearn.metrics.pairwise import cosine_similarity

# # Compute similarity matrix
# similarity_matrix = cosine_similarity(embeddings)

# # Print similarity of first two images
# print(f"Similarity between image 1 and 2: {similarity_matrix[0,1]}")
# print(f"Similarity between image 1 and 10: {similarity_matrix[0,10]}")
# print(f"Similarity between image 1 and 20: {similarity_matrix[0,20]}")
# print(f"Similarity between image 1 and 30: {similarity_matrix[5,200]}")


# print(f"Min: {embeddings.min()}, Max: {embeddings.max()}, Mean: {embeddings.mean()}, Std: {embeddings.std()}")

# # Reduce to 2D for visualization
# tsne = TSNE(n_components=2, perplexity=30, random_state=42)
# embeddings_2d = tsne.fit_transform(embeddings)

# # Plot clusters
# plt.figure(figsize=(10, 8))
# scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=label_color_mappings, cmap='jet', alpha=0.7)
# plt.colorbar(scatter, label="Identity")
# plt.title("Face Embedding Clusters (t-SNE)")
# plt.show()
