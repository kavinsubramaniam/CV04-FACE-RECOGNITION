import os
import cv2
from deepface import DeepFace
# import numpy as np
import pickle
from tqdm import tqdm

path = "../test/test_emd"

files = os.listdir(path)
images = []
labels = []
for label in files:
    image_folder_path = os.listdir(os.path.join(path, label))
    for idx in tqdm(range(len(image_folder_path)), desc=f"Importing the images->label: {label}"):
        # print(os.path.join(os.path.join(path, label), image_file))
        images.append(cv2.imread(os.path.join(os.path.join(path, label), image_folder_path[idx])))
        labels.append(label)

embeddings = []
print(labels)
print("Total number of images %d"%(len(images)))
model_name = ["Facenet", "Facenet512","OpenFace", "DeepFace", "DeepID", "Dlib", "ArcFace", "SFace","GhostFaceNet"]
model_name_f = model_name[1]
for idx in tqdm(range(len(images)), desc="Embedding the images"):
    # emd = DeepFace.represent(images[idx], detector_backend='skip', enforce_detection=False)
    emd = DeepFace.represent(images[idx],model_name=model_name_f, detector_backend='skip', enforce_detection=False)
    embeddings.append(emd)

with open(f"face_embedding_{model_name_f}.pkl", "wb") as file:
    pickle.dump(embeddings, file)

with open("label.pkl", "wb") as file:
    pickle.dump(labels, file)

# print(labels)