# Procrastination Pulverizer

Procrastination Pulverizer is an object detection system built using **YOLOv11** and OpenCV in Python. The project detects objects in a live video feed and triggers an alert when a mobile phone is detected, helping users stay focused.

## Features
- Real-time object detection using **YOLOv11**
- Alerts the user when a mobile phone is detected
- Graphical User Interface (GUI) using **Tkinter**
- Optimized threading for smooth video processing
- Auto-detects camera index

## Installation
### Prerequisites
Ensure you have **Python 3.8+** installed on your system. Then, install the required dependencies:
```bash
pip install ultralytics opencv-python numpy pillow tkinter
```

## Usage
Run the following command to start the object detection application:
```bash
python main.py
```

## Project Structure
```
procrastination_pulverizer/
│-- main.py              # Main application script
│-- yolo11x.pt           # YOLO model file (place it here after downloading)
│-- README.md            # Project documentation
```

## How It Works
1. The script captures frames from the webcam.
2. YOLOv8 processes the frames to detect objects.
3. If a **mobile phone** is detected, an alert popup appears saying **STOP**.
4. The application ensures smooth real-time processing with threading optimizations.

[YOLO Model (yolo11x.pt)](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x.pt)
