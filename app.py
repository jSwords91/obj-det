import argparse
from typing import Optional

from src.object_detection import ObjectDetector
from src.video_processor import VideoProcessor

def main(source: str, output: Optional[str] = None, input_type: Optional[str] = None, model: str = "/models/yolov10s.pt") -> None:
    """
    Main function to run the object detection and tracking application.

    Args:
        source (str): Path to the input video file or '0' for webcam.
        output (Optional[str]): Path to save the output video file. If None, display output in real-time.
        input_type (Optional[str]): Type of input - 'video', 'webcam', or 'image'. If None, determine based on source.
    """
    detector = ObjectDetector(model)
    processor = VideoProcessor(detector)

    if source == '0':
        input_type = 'webcam'
    elif input_type is None:
        input_type = 'video'

    if input_type == 'webcam':
        processor.process_webcam(output)
    elif input_type == 'image':
        processor.process_image(source, output)
    else:
        processor.process_video(source, output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Object Detection and Tracking App")
    parser.add_argument("--source", type=str, default='0', help="Path to input video file or '0' for webcam")
    parser.add_argument("--output", type=str, help="Path to save the output video file")
    parser.add_argument("--input_type", type=str, choices=['video', 'webcam', 'image'], help="Type of input - 'video', 'webcam', or 'image'")
    parser.add_argument("--model", type=str, default="/models/yolov10s.pt", help="Path to the YOLOvX model weights")
    args = parser.parse_args()

    main(args.source, args.output, args.input_type, args.model)