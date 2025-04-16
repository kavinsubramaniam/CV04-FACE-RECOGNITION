import onnxruntime as ort
# from deepface import DeepFace
# import faiss
# import os
import cv2
from typing import List
# import threading
import numpy as np


class FaceDetectionError(Exception):
    """Custom exception for Face recognition errors."""

    pass


class FaceDetection:
    """Class to perform face detection."""

    __detection_model: ort = None
    __input_size: int = 640

    def __init__(self, detection_model_path: str, input_size: int = 640):
        """
        Initialize the FaceDetection class.

        args:
            detection_model_path: str: Path to the face detection model.
            input_size: int: Input size for the model.

        """
        try:
            # Load the detection model
            self.__detection_model = ort.InferenceSession(detection_model_path)
            self.__input_size = input_size
        except Exception as e:
            raise FaceDetectionError(f"Failed to load detection model: {e}")

    def __onnx_output_processor(
        self,
        outputs: List[np.ndarray],
        original_w_h: tuple,
        conf_threshold: float,
    ) -> List[np.ndarray]:
        """
        Process the output from the ONNX model.

        args:
            output: List[np.ndarray]: Output from the ONNX model.

        returns:
            (x1, y1, x2, y2, confidence, distance): tuple : detected face.

        """

        original_w, original_h = original_w_h
        image_center_x, image_center_y = original_w / 2, original_h / 2  # Image center

        # Extract predictions and reshape
        predictions = outputs[0].squeeze(0).T  # Shape: (8400, 5)

        # Extract bounding boxes and confidence scores
        x_centers, y_centers, widths, heights, confidences = (
            predictions[:, 0],
            predictions[:, 1],
            predictions[:, 2],
            predictions[:, 3],
            predictions[:, 4],
        )

        # Filter detections with confidence threshold
        valid_indices = np.where(confidences > conf_threshold)[0]

        # Convert YOLO format (center-x, center-y, width, height) to OpenCV format (x1, y1, x2, y2)
        boxes = []
        for i in valid_indices:
            x_center, y_center, width, height, confidence = (
                x_centers[i],
                y_centers[i],
                widths[i],
                heights[i],
                confidences[i],
            )

            # Scale back to original image size
            x_center *= original_w / self.__input_size
            y_center *= original_h / self.__input_size
            width *= original_w / self.__input_size
            height *= original_h / self.__input_size

            # Convert to (x1, y1, x2, y2)
            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)

            # Calculate distance from image center
            distance = np.sqrt((x_center - image_center_x) ** 2 + (y_center - image_center_y) ** 2)
            boxes.append((x1, y1, x2, y2, confidence, distance))

        # Return the face with the minimum distance from the center
        return min(boxes, key=lambda b: b[5]) if boxes else ()

    def detect_faces(
        self, frame: np.ndarray, conf_threshold: float = 0.85,
    ) -> List[np.ndarray]:
        """
        Detect faces in the input frame.
        
        args: 
            frame: np.ndarray: Input frame.
            conf_threshold: float: Confidence threshold for detections.
            
        returns:
            (x1, y1, x2, y2, confidence, distance): tuple : detected face.
        
        """
        try:
            # Get original image size
            original_h, original_w = frame.shape[:2]
        except Exception as e:
            raise FaceDetectionError(f"Failed to get frame size maybe it is None: {e}")

        # Resize while maintaining aspect ratio
        img_resized = cv2.resize(frame, (self.__input_size, self.__input_size))
        input_tensor = (
            img_resized.transpose(2, 0, 1).astype(np.float32)[np.newaxis, ...] / 255.0
        )  # Normalize

        try: 
            # Get input name for ONNX model
            input_name = self.__detection_model.get_inputs()[0].name
        
            # Run inference
            outputs = self.__detection_model.run(None, {input_name: input_tensor})

            # Process the output
            face = self.__onnx_output_processor(
                outputs, (original_w, original_h), conf_threshold
            )
        
        except Exception as e:
            raise FaceDetectionError(f"Failed to detect faces: {e}")

        return face if face else ()

    

if __name__ == "__main__":
    # Load face detection model
    detection_model_path = "../models/yolov11n-face.onnx"
    face_detection = FaceDetection(detection_model_path)

    # Start webcam
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break

        # Detect faces
        face = face_detection.detect_faces(frame)

        if face:
            # Face bounding box
            x1, y1, x2, y2, confidence, distance = face
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Display the frame
        cv2.imshow("Face Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows


