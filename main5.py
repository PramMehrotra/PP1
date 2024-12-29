import cv2
import time
import threading
import queue
from ultralytics import YOLO

# Initialize YOLO model
model = YOLO('yolov8n.pt')  # You can replace 'yolov8n.pt' with your model path

# Queue for holding frames
frame_queue = queue.Queue(maxsize=10)

# Flag for controlling the threads
stop_threads = False

# Frame capturing thread
def capture_frames(cap):
    global stop_threads
    while not stop_threads:
        ret, frame = cap.read()
        if ret:
            if not frame_queue.full():
                frame_queue.put(frame)  # Put the captured frame in the queue

# Object detection thread
def process_frames():
    global stop_threads
    prev_time = time.time()
    while not stop_threads:
        if not frame_queue.empty():
            frame = frame_queue.get()

            # Perform object detection using YOLOv8
            results = model(frame)  # Inference

            # Parse the results
            for result in results:
                boxes = result.boxes  # Detection boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Coordinates of the bounding box
                    confidence = box.conf[0]  # Confidence score
                    label = model.names[int(box.cls[0])]  # Class label

                    if confidence > 0.5:  # Confidence threshold
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f'{label} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Calculate FPS
            current_time = time.time()
            elapsed_time = current_time - prev_time
            fps = 1 / elapsed_time
            prev_time = current_time

            cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            # Display the output
            cv2.imshow("Webcam", frame)

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Set resolution
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)  # Request 30 FPS

# Start the threads
capture_thread = threading.Thread(target=capture_frames, args=(cap,))
process_thread = threading.Thread(target=process_frames)

capture_thread.start()
process_thread.start()

try:
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_threads = True  # Stop threads if 'q' is pressed
            break
finally:
    stop_threads = True  # Ensure that threads stop
    capture_thread.join()
    process_thread.join()
    cap.release()
    cv2.destroyAllWindows()
