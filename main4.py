import cv2
import time
import threading
import queue
from ultralytics import YOLO
import tkinter as tk
from tkinter import messagebox

# Initialize YOLO model
model = YOLO('yolov8n.pt')  # Replace with your model path if needed

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
                frame_queue.put(frame)
            else:
                frame_queue.get()
                frame_queue.put(frame)

# Function to show pop-up
def show_stop_popup():
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    messagebox.showwarning("Alert", "STOP")
    root.after(3000, root.destroy)  # Close the pop-up after 3 seconds
    root.mainloop()

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
            phone_detected = False
            for result in results:
                boxes = result.boxes  # Detection boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Coordinates of the bounding box
                    confidence = box.conf[0]  # Confidence score
                    label = model.names[int(box.cls[0])]  # Class label

                    if confidence > 0.5:  # Confidence threshold
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f'{label} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                        # Check if the detected object is a phone
                        if label.lower() in ["cell phone", "phone"]:
                            phone_detected = True

            # Trigger pop-up if a phone is detected
            if phone_detected:
                threading.Thread(target=show_stop_popup, daemon=True).start()

            # Calculate FPS
            current_time = time.time()
            elapsed_time = current_time - prev_time
            fps = 1 / elapsed_time
            prev_time = current_time

            cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            # Put processed frame back into queue for display
            if not frame_queue.full():
                frame_queue.put(frame)

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
        if not frame_queue.empty():
            # Get processed frame from queue
            frame = frame_queue.get()

            # Display the output
            cv2.imshow("Webcam", frame)

            # Check for key press (non-blocking)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_threads = True  # Stop threads if 'q' is pressed
                break
finally:
    stop_threads = True  # Ensure that threads stop
    capture_thread.join()
    process_thread.join()
    cap.release()
    cv2.destroyAllWindows()
