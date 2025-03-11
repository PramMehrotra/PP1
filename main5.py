import cv2
import time
import threading
import queue
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
from ultralytics import YOLO

# Initialize YOLO model with improved parameters
model = YOLO('yolov8n.pt')

# Queue for holding frames
frame_queue = queue.Queue(maxsize=8)

# Flag to control threads
stop_threads = threading.Event()
capture_thread = None
process_thread = None
cap = None
popup_open = False

# Function to auto-detect camera index
def get_camera_index():
    for i in range(5):
        temp_cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if temp_cap.isOpened():
            temp_cap.release()
            return i
    return -1

# Create the Tkinter window
root = tk.Tk()
root.title("Object Detection UI")
root.geometry("900x600")

title_label = ttk.Label(root, text="Object Detection with YOLOv8", font=("Helvetica", 16, "bold"))
title_label.pack(pady=10)

# Adjust frame settings for better resizing
video_frame = ttk.LabelFrame(root, text="Live Feed", padding=10)
video_frame.pack(pady=10, padx=10, fill="both", expand=True)

label_img = ttk.Label(video_frame)
label_img.pack(fill="both", expand=True)  # Ensure label resizes properly

# Show pop-up when a phone is detected 
def show_stop_popup():
    global popup_open
    if popup_open:
        return
    popup_open = True
    popup = tk.Toplevel(root)
    popup.title("Alert")
    popup.geometry("300x150")
    popup.resizable(False, False)

    label = ttk.Label(popup, text="STOP", font=("Helvetica", 18, "bold"), foreground="red")
    label.pack(pady=20)

    def close_popup():
        global popup_open
        popup_open = False
        popup.destroy()

    dismiss_btn = ttk.Button(popup, text="Dismiss", command=close_popup)
    dismiss_btn.pack(pady=10)

# Capture frames efficiently
def capture_frames():
    global cap
    camera_index = get_camera_index()
    if camera_index == -1:
        messagebox.showerror("Error", "No camera detected!")
        return

    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    while not stop_threads.is_set():
        ret, frame = cap.read()
        if ret and not frame_queue.full():
            frame_queue.put_nowait(frame)

    if cap:
        cap.release()

# Object detection thread with improved efficiency
def process_frames():
    prev_time = time.time()
    frame_count = 0
    fps_target = 15  # Adjust for real-time processing

    while not stop_threads.is_set():
        try:
            frame = frame_queue.get_nowait()
        except queue.Empty:
            continue

        frame_count += 1
        if frame_count % (30 // fps_target) != 0:
            continue  # Skip frames dynamically based on FPS target
        
        results = model.predict(frame, verbose=False, conf=0.5, imgsz=736, agnostic_nms=True, iou=0.5)
        
        phone_detected = any(
            model.names[int(box.cls[0])].lower() in ["cell phone", "phone", "mobile"]
            for result in results for box in result.boxes if box.conf[0] > 0.5
        )

        if phone_detected:
            root.after(0, show_stop_popup)

        current_time = time.time()
        fps = int(1 / (current_time - prev_time))
        prev_time = current_time

        cv2.putText(frame, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        if not frame_queue.full():
            frame_queue.put_nowait(frame)

# Update UI efficiently
def update_ui():
    if stop_threads.is_set():
        label_img.config(image='')  # Clear UI on stop
        return
    
    try:
        frame = frame_queue.get_nowait()
    except queue.Empty:
        root.after(30, update_ui)
        return

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (700, 500))
    img = ImageTk.PhotoImage(Image.fromarray(frame))
    label_img.config(image=img)
    label_img.image = img

    root.after(30, update_ui)  # Slightly reduced update frequency

# Start detection
def start_detection():
    global capture_thread, process_thread

    if capture_thread and capture_thread.is_alive():
        return

    stop_threads.clear()
    while not frame_queue.empty():
        frame_queue.get_nowait()

    capture_thread = threading.Thread(target=capture_frames, daemon=True)
    process_thread = threading.Thread(target=process_frames, daemon=True)

    capture_thread.start()
    process_thread.start()
    update_ui()

# Stop detection and release resources
def stop_detection():
    global cap
    stop_threads.set()

    if cap:
        cap.release()
        cap = None

    while not frame_queue.empty():
        frame_queue.get_nowait()

    label_img.config(image='')

# Buttons for control
button_frame = ttk.Frame(root)
button_frame.pack(pady=10, fill="x")  # Allow frame to resize properly

start_btn = ttk.Button(button_frame, text="Start Detection", command=start_detection)
start_btn.grid(row=0, column=0, padx=10, sticky="ew")  # Make button expand

stop_btn = ttk.Button(button_frame, text="Stop Detection", command=stop_detection)
stop_btn.grid(row=0, column=1, padx=10, sticky="ew")  # Make button expand

button_frame.columnconfigure(0, weight=1)  # Ensure equal button width
button_frame.columnconfigure(1, weight=1)

# Start Tkinter event loop
root.mainloop()
