import faiss
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from deepface import DeepFace
import numpy as np
from typing import List
import cv2
from tqdm import tqdm
import pickle

class FaceEmbeddingsError(Exception):
    """Custom exception for Face Embeddings errors."""
    pass

class FaceEmbeddings:
    """Class to perform face embeddings."""

    __path: str = "./face_embeddings/"
    __index: faiss.Index = None
    __data: dict = dict()

    def __init__(self, path):
        self.__path = path

    def __check_index(self):
        if not self.__index:
            raise FaceEmbeddingsError("Index not created (or) read from the memory.")

    def create_index(self, member_count:int=10, shape:int = 4096, m:int = 8, bits:int = 8) -> None:
        """
        Create a Faiss index for face embeddings.

        args:
            member_count: int: Number of members in the index.
            shape: int: Shape of the face embeddings.
            m: int: Number of subquantizers.
            bits: int: Number of bits per subquantizer.
        
        returns:
            index: faiss index: Faiss index for face embeddings.
        """
        # quantizer = faiss.IndexFlatL2(shape)  # L2 distance quantizer
        # index = faiss.IndexIVFPQ(quantizer, shape, member_count, m, bits)
        index = faiss.IndexFlatL2(shape)
        self.__index = index
    
    def read_index(self) -> None:
        """
        Read the index from the file.
        """
        try:
            # Load the index from the file
            if not os.path.exists(os.path.join(self.__path, f"faiss_index.index")):
                self.create_index()
            else:
                self.__index = faiss.read_index(os.path.join(self.__path, f"faiss_index.index"))
            
            # Load or Initialize User Mappings
            if os.path.exists(os.path.join(self.__path, f"user_mappings.pkl")):
                with open(os.path.join(self.__path, f"user_mappings.pkl"), "rb") as f:
                    self.__data = pickle.load(f)
            else:
                self.__data = {}

        except Exception as e:
            raise FaceEmbeddingsError(f"Failed to load index: {e}")

    def write_index(self) -> None:
        """
        Write the index to the file.
        """
        self.__check_index()
        # Ensure the directory exists
        index_dir = os.path.join(self.__path, "faiss_index.index")
        os.makedirs(os.path.dirname(index_dir), exist_ok=True)

        # Save user mappings
        with open(os.path.join(self.__path, f"user_mappings.pkl"), "wb") as f:
            pickle.dump(self.__data, f)

        # Write the FAISS index
        faiss.write_index(self.__index, index_dir)
    
    def insert(self, embeddings: List[np.ndarray], name: str) -> None:
        """
        Insert face embeddings into the index.

        args:
            embeddings: List[np.ndarray]: List of Face embeddings to insert.
        """
        self.__check_index()
        embeddings = np.vstack(embeddings).astype(np.float32)  # Ensure 2D array
        if not self.__index.is_trained:
            self.__index.train(embeddings)  # Train on all embeddings
        start_idx = self.__index.ntotal-1
        self.__index.add(embeddings)  # Add embeddings

        if name in self.__data.keys():
            self.__data[name] = self.__data[name].union(set(range(start_idx, self.__index.ntotal-1)))
        else:
            self.__data[name] = set(range(start_idx, self.__index.ntotal-1))

    def __search_mappings(self, ids):
        # Count the number of matches for each user
        match_counts = {user: len(values.intersection(ids)) for user, values in self.__data.items()}

        # Get the user with the most matches
        best_match = max(match_counts, key=match_counts.get)
        return best_match
    
    def search_embedding(self, embedding: np.ndarray, k:int = 1, threshold:float = 1.2):
        self.__check_index()
        if self.__index.ntotal == 0:
            raise FaceEmbeddingsError("No embeddings in index to search.")
        distance, index_ids = self.__index.search(embedding, k)
        best_match_distance = distance[0][0]
        best_match_index = index_ids[0][0]
        # print(distance)
        # print(index_ids)
        # print(self.__data)

        # If no valid match found, return "Unknown"
        if best_match_index == -1 or best_match_distance > threshold:
            return f"Unknown, {best_match_distance, index_ids[0]}"
        print(f"{best_match_distance, index_ids[0]}")
        index_ids = index_ids[0]
       
        return self.__search_mappings(index_ids)

    def search_face(self, face_image: np.ndarray, k: int = 1, backend_model:str = 'VGG-Face', threshold:float = 1.2) -> tuple:
        """
        Search for a face in the index.

        args:
            embeddings: np.ndarray: Face embeddings to search.
            k: int: Number of neighbors to return.

        returns:
            (distances, indices): tuple: Nearest neighbors.
        """
        self.__check_index()
        if self.__index.ntotal == 0:
            raise FaceEmbeddingsError("No embeddings in index to search.")
        embedding = np.array(DeepFace.represent(face_image, model_name=backend_model, detector_backend='skip', enforce_detection=False)[0]['embedding'])
        embedding = embedding.reshape(1, -1).astype(np.float32)
        distance, index_ids = self.__index.search(embedding, k)
        best_match_distance = distance[0][0]
        best_match_index = index_ids[0][0]

        # If no valid match found, return "Unknown"
        if best_match_index == -1 or best_match_distance > threshold:
            return f"Unknown, {best_match_distance, index_ids[0]}"
        print(f"{best_match_distance, index_ids[0]}")
        index_ids = index_ids[0]
       
        return self.__search_mappings(index_ids)
    
    def create_and_insert_embeddings(self, face_imgs: List[np.ndarray], id: int,backend_model: str = 'VGG-Face') -> None:
        """
        Create and insert face embeddings into the index.

        args:
            face_imgs: List[np.ndarray]: Face images in form of np.ndarray list.
        """
        embeddings = []
        
        for image in tqdm(face_imgs, desc="Generating Embeddings", unit="image"):
            try:
                embedding_result = DeepFace.represent(image, model_name=backend_model, detector_backend='skip', enforce_detection=False)
                if embedding_result:
                    embeddings.append(np.array(embedding_result[0]['embedding']))
                else:
                    print("Warning: No embedding generated for an image.")
            except Exception as e:
                print(f"Error processing an image: {e}")

        if not embeddings:
            raise ValueError("DeepFace failed to generate embeddings for all images.")

        self.insert(embeddings, id)
    
    @property
    def index(self) -> faiss.Index:
        """
        Get the Faiss index.
        """
        self.__check_index()
        return self.__index
    
    @property
    def path(self) -> str:
        """
        Get the path for the face embeddings.
        """
        return self.__path

    @path.setter
    def path(self, path: str) -> None:
        """
        Set the path for the face embeddings.
        """
        self.__path = path
    
if __name__ == "__main__":
    # from face_detection import FaceDetection
    # detection_model_path = "../models/yolov11n-face.onnx"
    # face_detection = FaceDetection(detection_model_path)
    # name = "kavin"
    # path = f"../test/{name}"
    # if not os.path.exists(path):
    #     os.makedirs(path)
    # cap = cv2.VideoCapture(0)
    # count = 500
    # face_images = []
    # while count>0:
    #     ret, frame = cap.read() 

    #     if not ret:
    #         print("Failed to capture frame.")
    #         break

    #     # Detect faces
    #     face = face_detection.detect_faces(frame)

    #     if face:
    #         x1, y1, x2, y2, _, _ = face
    #         face_img = frame[y1:y2, x1:x2]
    #         face_images.append(face_img)
    #         cv2.imwrite(os.path.join(path, f"face_{count}.jpg"), face_img)
    #         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    #         count -= 1
        
    #     cv2.imshow("Frame", frame)

    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break

    # cap.release()
    # cv2.destroyAllWindows()

    # face_embeddings = FaceEmbeddings("../face_embeddings/")
    # face_embeddings.create_index()
    # # face_embeddings.read_index()
    # face_embeddings.create_and_insert_embeddings(face_images, name)
    # face_embeddings.write_index()


    # # frame = cv2.imread('../test/rash/face_0.jpg')
    # face_embeddings = FaceEmbeddings("../face_embeddings/")
    # face_embeddings.read_index()
    # print(face_embeddings.index.ntotal)

    # print(f"{face_embeddings.search_face(frame, k=10)}")

    face_embeddings = FaceEmbeddings("../face_embeddings/")
    face_embeddings.read_index()
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
            name = face_embeddings.search_face(frame, k=10, threshold=1.55)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            print(f"{name}")
            
        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # if os.path.exists("../face_embeddings/user_mappings.pkl"):
    #     with open("../face_embeddings/user_mappings.pkl", "rb") as f:
    #         data = pickle.load(f)
    # print(data)


    # DeepFace.stream(db_path="../test/find/", detector_backend='skip', enable_face_analysis=False)
    # result = DeepFace.verify(img1_path = "../test/face_0.jpg", img2_path = "../test/find/rash/face_1.jpg")
    # print(result)