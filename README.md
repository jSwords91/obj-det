# Object Detection and Tracking

## Bletchley Talk Idea... ##

This project is a simple object detection and tracking application built using Python and the Ultralytics YOLOvN models. It allows you to detect and track objects in real-time video streams or saved video files.

## Features

- **Object Detection**: Utilizes YOLO for real-time object detection (several variants available)
- **Object Tracking**: Implements a simple tracking system to follow detected objects.
- **Video Processing**: Supports processing video files and live webcam feeds.

## Usage

### Prerequisites

From terminal:

```bash
pip install -r requirements.txt
```

### Run the app from the terminal

```bash
python app.py
```

### Optional arguments

```bash
python app.py --source 0 --output output.mp4 --input_type webcam --model /models/yolov10s.pt
```

Or to use the video sample

```bash
python app.py --source carTest.mp4 --input_type video --output carOut.mp4
```

