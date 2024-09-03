from typing import List, Tuple, Dict
import cv2
import numpy as np
import os
from ultralytics import YOLO
import torch

class ObjectDetector:
    def __init__(self, model_path: str = "/models/yolov10s.pt") -> None:
        """
        Initialize the ObjectDetector with a YOLOvN model. If the model does not exist locally, download it.

        Args:
            model_path (str): Path to the YOLOvN model weights.
        """
        self.model = YOLO(model_path)
    
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            self.model = self.model.to(device)
        
        print("Using model: ", model_path, " on device: ", device)

    def detect_and_track(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """
        Perform object detection and tracking on a single frame.

        Args:
            frame (np.ndarray): Input frame for detection and tracking.

        Returns:
            Tuple[np.ndarray, List[Dict]]: Annotated frame and list of detection results.
        """
        results = self.model.track(frame, persist=True)
        annotated_frame = results[0].plot()
        return annotated_frame, results[0].boxes.data.tolist()

    @staticmethod
    def draw_labels(frame: np.ndarray, detections: List[Dict], color: Tuple[int, int, int] = (0, 255, 0), thickness: int = 2, font_scale: float = 0.5) -> np.ndarray:
        """
        Draw labels and bounding boxes on the frame.

        Args:
            frame (np.ndarray): Input frame to draw on.
            detections (List[Dict]): List of detection results.
            color (Tuple[int, int, int]): Color of the bounding box and text.
            thickness (int): Thickness of the bounding box.
            font_scale (float): Font scale for the text.

        Returns:
            np.ndarray: Frame with labels and bounding boxes drawn.
        """
        for det in detections:
            # handle case where track_id is not there
            x1, y1, x2, y2, conf, cls, track_id = det if len(det) == 7 else (*det, None)
            #id_class_label = f"ID: {int(track_id)} Class: {int(cls)}"
            #conf_label = f"Conf: {conf:.2f}"
            
            # Draw bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness, cv2.LINE_AA)
            
        return frame