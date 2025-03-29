# Procrastination Pulverizer

Procrastination Pulverizer is an advanced real-time object detection system designed to help users stay focused by detecting mobile phones and monitoring face attention using **YOLOv11** and OpenCV in Python. The project alerts users when distractions are detected and logs detection data for analysis.

## Features
- **Real-time Object Detection**: Uses **YOLOv11** for accurate detection of mobile phones and faces.
- **Distraction Alerts**: Pop-up warnings when a mobile phone is detected or the user looks away for too long.
- **Detection Logging**: Keeps track of total detections and session runtime.
- **Graphical User Interface (GUI)**: Built using **Tkinter** for easy user interaction.
- **Efficient Video Processing**: Optimized threading for smooth performance.
- **Auto Camera Detection**: Automatically selects an available webcam.
- **Enhanced FPS Display**: Shows real-time FPS for performance monitoring.

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
│-- phone_detections.txt # Log file for detection history
│-- README.md            # Project documentation
```

## How It Works
1. The script captures frames from the webcam.
2. **YOLOv11** processes frames to detect objects with high accuracy.
3. If a **mobile phone** is detected, a pop-up alert appears with a **STOP warning**.
4. If the user **looks away** for too long, another pop-up appears with a **Focus Warning**.
5. The application ensures real-time processing with optimized threading and efficient frame handling.
6. Detection history is logged, tracking the number of phone detections and session runtime.

[Download YOLO Model (yolo11x.pt)](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x.pt)

