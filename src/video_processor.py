import signal
from typing import Optional, List, Dict
import cv2
from .object_detection import ObjectDetector


class VideoProcessor:
    def __init__(self, detector: ObjectDetector) -> None:
        """
        Initialize the VideoProcessor with an ObjectDetector.

        Args:
            detector (ObjectDetector): An instance of the ObjectDetector class.
        """
        self.detector = detector
        self.cap = None
        self.out = None

        # Register signal handler for graceful termination
        signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, sig, frame) -> None:
        """
        Signal handler for graceful termination.
        """
        print("Interrupt received, stopping...")
        if self.cap:
            self.cap.release()
        if self.out:
            self.out.release()
        cv2.destroyAllWindows()
        exit(0)

    def process_video(self, input_path: str, output_path: Optional[str] = None) -> None:
        """
        Process a video file for object detection and tracking.

        Args:
            input_path (str): Path to the input video file.
            output_path (Optional[str]): Path to save the output video file. If None, display output in real-time.
        """
        self.cap = cv2.VideoCapture(input_path)
        if not self.cap.isOpened():
            raise ValueError(f"Error opening video file: {input_path}")

        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))

        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            annotated_frame, detections = self.detector.detect_and_track(frame)
            final_frame = self.detector.draw_labels(annotated_frame, detections)

            if output_path:
                self.out.write(final_frame)
            else:
                cv2.imshow("Object Detection and Tracking", final_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        self.cap.release()
        if output_path:
            self.out.release()
        cv2.destroyAllWindows()

    def process_webcam(self, output_path: Optional[str] = None) -> None:
        """
        Process webcam input for object detection and tracking.

        Args:
            output_path (Optional[str]): Path to save the output video file. If None, display output in real-time.
        """
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise ValueError("Error opening webcam")

        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = 30

        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') # fourcc is a 4-character code
            self.out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            annotated_frame, detections = self.detector.detect_and_track(frame)
            final_frame = self.detector.draw_labels(annotated_frame, detections)

            if output_path:
                self.out.write(final_frame)
            else:
                cv2.imshow("Object Detection and Tracking", final_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        self.cap.release()
        if output_path:
            self.out.release()
        cv2.destroyAllWindows()

    def process_image(self, input_path: str, output_path: Optional[str] = None) -> None:
        """
        Process an image file for object detection and tracking.

        Args:
            input_path (str): Path to the input image file.
            output_path (Optional[str]): Path to save the output image file. If None, display output in real-time.
        """
        frame = cv2.imread(input_path)
        if frame is None:
            raise ValueError(f"Error opening image file: {input_path}")

        annotated_frame, detections = self.detector.detect_and_track(frame)
        final_frame = self.detector.draw_labels(annotated_frame, detections)

        if output_path:
            cv2.imwrite(output_path, final_frame)
        else:
            cv2.imshow("Object Detection and Tracking", final_frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()