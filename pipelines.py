from src.face_embedding import FaceEmbeddings
import os
import cv2

class FaceRecognitionPipeline:
    face_embeddings = FaceEmbeddings()


    def __init__(self):
        if os.path.exists(self.face_embeddings.path):
            self.face_embeddings.read_index()
        else:
            self.face_embeddings.create_index()
        print("successfully read the index", self.face_embeddings.index.ntotal)

    def register(self, face_images):
        """
        Register the pipeline.
        """
        self.face_embeddings.create_and_insert_embeddings(face_images)
        self.face_embeddings.write_index()

    def recognize(self, face):
        """
        Recognize face.
        """
        return self.face_embeddings.search_face(face, k=10)


if __name__ == "__main__":
    face_recognition_pipeline = FaceRecognitionPipeline()
    image_path = "./test/rash/face_0.jpg"
    frame = cv2.imread(image_path)
    index = face_recognition_pipeline.recognize(frame)
    print(index)
    
