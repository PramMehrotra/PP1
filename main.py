import cv2
import time
import threading
import queue
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
from ultralytics import YOLO
import numpy as np

# Initialize YOLO model with improved parameters
model = YOLO('yolo11x.pt')

# Queue for holding frames
frame_queue = queue.Queue(maxsize=8)  # Reduced queue size for less delay

# Flag to control threads
stop_threads = threading.Event()
capture_thread = None
process_thread = None
cap = None

# File to store detection counts and runtime
DETECTION_LOG_FILE = "phone_detections.txt"

# Popup tracking variables
phone_popup_open = False
distraction_popup_open = False
popup_windows = []  # List to track all popup windows
phone_detection_count = 0  # Counter for phone detections
total_detections = 0  # Total detections across all sessions
total_runtime = 0  # Total runtime across all sessions
session_start_time = None  # Start time of current session

def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    remaining_seconds = int(seconds % 60)
    return f"{hours}h {minutes}min {remaining_seconds}sec"

def parse_time(time_str):
    try:
        # Split the time string into components
        parts = time_str.split()
        hours = int(parts[0].replace('h', ''))
        minutes = int(parts[1].replace('min', ''))
        seconds = int(parts[2].replace('sec', ''))
        return hours * 3600 + minutes * 60 + seconds
    except:
        return 0

def load_detection_count():
    global total_detections, total_runtime
    try:
        with open(DETECTION_LOG_FILE, 'r') as f:
            lines = f.readlines()
            if len(lines) >= 2:
                total_detections = int(lines[0].strip().split(':')[1])
                total_runtime = parse_time(lines[1].strip().split(':')[1])
    except (FileNotFoundError, ValueError, IndexError):
        total_detections = 0
        total_runtime = 0

def save_detection_count():
    with open(DETECTION_LOG_FILE, 'w') as f:
        f.write(f"phone detections:{total_detections}\n")
        f.write(f"time:{format_time(total_runtime)}\n")

# Load previous detection count when program starts
load_detection_count()

# Head pose tracking variables
last_face_time = time.time()
face_detected = False
look_away_start = None
LOOK_AWAY_THRESHOLD = 3.0  # seconds

# Detection state variables
last_phone_detection = None
last_face_detection = None
DETECTION_PERSISTENCE = 0.3  # Reduced persistence time

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
root.minsize(900, 600)  # Set minimum window size

# Main container frame
main_frame = ttk.Frame(root)
main_frame.pack(fill="both", expand=True, padx=10, pady=10)

title_label = ttk.Label(main_frame, text="Object Detection with YOLO11x", font=("Helvetica", 16, "bold"))
title_label.pack(pady=10)

# Video frame with fixed height
video_frame = ttk.LabelFrame(main_frame, text="Live Feed", padding=10)
video_frame.pack(pady=10, padx=10, fill="both", expand=True)

label_img = ttk.Label(video_frame)
label_img.pack(fill="both", expand=True)

# Button frame at bottom
button_frame = ttk.Frame(main_frame)
button_frame.pack(side="bottom", pady=10, fill="x")  # Pack at bottom and fill horizontally

# Show pop-up when a phone is detected 
def show_stop_popup():
    global phone_popup_open, phone_detection_count
    if phone_popup_open:
        return
    phone_popup_open = True
    popup = tk.Toplevel(root)
    popup.title("Phone Alert")
    popup.geometry("300x200")  # Made window taller to accommodate counter
    popup.resizable(False, False)
    
    # Position the popup based on existing popups
    x = root.winfo_x() + 50 + len(popup_windows) * 20
    y = root.winfo_y() + 50 + len(popup_windows) * 20
    popup.geometry(f"+{x}+{y}")

    # Create a frame for the alert content
    content_frame = ttk.Frame(popup)
    content_frame.pack(fill="both", expand=True, padx=10, pady=10)

    # Alert label
    label = ttk.Label(content_frame, text="STOP", font=("Helvetica", 24, "bold"), foreground="red")
    label.pack(pady=10)

    # Counter frame with border
    counter_frame = ttk.Frame(content_frame, relief="solid", borderwidth=2)
    counter_frame.pack(pady=10, padx=10, fill="x")

    # Counter label with larger font
    counter_label = ttk.Label(counter_frame, text=f"Phone Detections: {phone_detection_count}", 
                             font=("Helvetica", 14, "bold"))
    counter_label.pack(pady=5)

    def update_counter():
        if phone_popup_open:  # Only update if popup is still open
            counter_label.config(text=f"Phone Detections: {phone_detection_count}")
            popup.after(100, update_counter)  # Update every 100ms

    def close_popup():
        global phone_popup_open, distraction_popup_open
        phone_popup_open = False
        distraction_popup_open = False
        # Close all popup windows
        for window in popup_windows:
            window.destroy()
        popup_windows.clear()

    dismiss_btn = ttk.Button(content_frame, text="Dismiss", command=close_popup)
    dismiss_btn.pack(pady=10)
    
    # Add to popup windows list
    popup_windows.append(popup)
    
    # Start counter updates
    update_counter()

# Show pop-up when distracted
def show_distraction_popup():
    global distraction_popup_open
    if distraction_popup_open:
        return
    distraction_popup_open = True
    popup = tk.Toplevel(root)
    popup.title("Distraction Alert")
    popup.geometry("300x150")
    popup.resizable(False, False)
    
    # Position the popup based on existing popups
    x = root.winfo_x() + 50 + len(popup_windows) * 20
    y = root.winfo_y() + 50 + len(popup_windows) * 20
    popup.geometry(f"+{x}+{y}")

    label = ttk.Label(popup, text="STAY FOCUSED!", font=("Helvetica", 18, "bold"), foreground="red")
    label.pack(pady=20)

    def close_popup():
        global phone_popup_open, distraction_popup_open
        phone_popup_open = False
        distraction_popup_open = False
        # Close all popup windows
        for window in popup_windows:
            window.destroy()
        popup_windows.clear()

    dismiss_btn = ttk.Button(popup, text="Dismiss", command=close_popup)
    dismiss_btn.pack(pady=10)
    
    # Add to popup windows list
    popup_windows.append(popup)

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
    global face_detected, look_away_start, last_face_time, last_phone_detection, last_face_detection, phone_detection_count, total_detections
    prev_time = time.time()
    frame_count = 0
    fps_target = 20  # Increased FPS target

    while not stop_threads.is_set():
        try:
            frame = frame_queue.get_nowait()
        except queue.Empty:
            continue

        frame_count += 1
        if frame_count % (30 // fps_target) != 0:
            continue  # Skip frames dynamically based on FPS target
        
        # Enhanced detection parameters
        results = model.predict(frame, verbose=False, conf=0.25, imgsz=736, agnostic_nms=True, iou=0.45)
        
        # Enhanced phone detection with more keywords and lower threshold
        phone_detected = False
        current_face_detected = False
        current_time = time.time()
        
        # Create a copy of the frame for drawing
        display_frame = frame.copy()
        
        # Process detections
        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = model.names[cls].lower()
                
                # Check for faces
                if class_name == "person" and conf > 0.5:
                    current_face_detected = True
                    last_face_time = current_time
                    last_face_detection = current_time
                    
                    # Get face box coordinates
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Draw face box with thicker lines
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    
                    # Add face label
                    cv2.putText(display_frame, "Face", (x1, y1 - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Check if face is looking at screen (simple heuristic)
                    face_center = (x1 + x2) // 2
                    frame_center = display_frame.shape[1] // 2
                    if abs(face_center - frame_center) > 100:  # If face is significantly off-center
                        if look_away_start is None:
                            look_away_start = current_time
                        # Draw warning text when looking away
                        cv2.putText(display_frame, "LOOKING AWAY!", (x1, y2 + 20), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    else:
                        look_away_start = None
                
                # Phone detection
                if any(keyword in class_name for keyword in ["phone", "mobile", "cell", "smartphone", "device", "handset"]):
                    phone_detected = True
                    last_phone_detection = current_time
                    phone_detection_count += 1  # Increment session counter
                    total_detections += 1  # Increment total counter
                    save_detection_count()  # Save updated total to file
                    
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Draw bounding box with thicker lines
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    
                    # Add label with confidence
                    label = f'Phone {conf:.2f}'
                    cv2.putText(display_frame, label, (x1, y1 - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    # Add warning text
                    cv2.putText(display_frame, "WARNING!", (x1, y2 + 20), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Persist detections for a short time
        if last_phone_detection and current_time - last_phone_detection < DETECTION_PERSISTENCE:
            phone_detected = True
        if last_face_detection and current_time - last_face_detection < DETECTION_PERSISTENCE:
            current_face_detected = True

        # Update face detection status
        if not current_face_detected and face_detected:
            if time.time() - last_face_time > LOOK_AWAY_THRESHOLD:
                root.after(0, show_distraction_popup)
        face_detected = current_face_detected

        # Check for prolonged look away
        if look_away_start is not None and time.time() - look_away_start > LOOK_AWAY_THRESHOLD:
            root.after(0, show_distraction_popup)
            look_away_start = None

        if phone_detected:
            root.after(0, show_stop_popup)

        fps = int(1 / (current_time - prev_time))
        prev_time = current_time

        # Enhanced FPS display
        cv2.putText(display_frame, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add detection status with background
        status = "Phone Detected!" if phone_detected else "No Phone Detected"
        color = (0, 0, 255) if phone_detected else (0, 255, 0)
        cv2.putText(display_frame, status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Add face detection status with background
        face_status = "Face Detected" if face_detected else "No Face Detected"
        face_color = (0, 255, 0) if face_detected else (0, 0, 255)
        cv2.putText(display_frame, face_status, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, face_color, 2)

        # Put the processed frame back in the queue
        if not frame_queue.full():
            frame_queue.put_nowait(display_frame)

# Update UI efficiently
def update_ui():
    if stop_threads.is_set():
        label_img.config(image='')  # Clear UI on stop
        return
    
    try:
        frame = frame_queue.get_nowait()
    except queue.Empty:
        root.after(30, update_ui)  # Reduced delay for faster updates
        return

    # Convert to RGB and resize
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (700, 500))
    
    # Convert to PhotoImage and update display
    img = ImageTk.PhotoImage(Image.fromarray(frame))
    label_img.config(image=img)
    label_img.image = img  # Keep a reference to prevent garbage collection

    root.after(30, update_ui)  # Reduced delay for faster updates

# Start detection
def start_detection():
    global capture_thread, process_thread, session_start_time
    if capture_thread and capture_thread.is_alive():
        return

    stop_threads.clear()
    while not frame_queue.empty():
        frame_queue.get_nowait()

    session_start_time = time.time()  # Record session start time
    capture_thread = threading.Thread(target=capture_frames, daemon=True)
    process_thread = threading.Thread(target=process_frames, daemon=True)

    capture_thread.start()
    process_thread.start()
    update_ui()

# Stop detection and release resources
def stop_detection():
    global cap, phone_popup_open, distraction_popup_open, popup_windows, phone_detection_count, total_runtime
    stop_threads.set()

    if cap:
        cap.release()
        cap = None

    while not frame_queue.empty():
        frame_queue.get_nowait()

    label_img.config(image='')
    
    # Close all popup windows
    for popup in popup_windows:
        popup.destroy()
    popup_windows.clear()
    phone_popup_open = False
    distraction_popup_open = False
    
    # Calculate and update runtime
    if session_start_time is not None:
        session_runtime = time.time() - session_start_time
        total_runtime += session_runtime
        save_detection_count()  # Save updated totals
    
    # Show total detections count before resetting
    if phone_detection_count > 0:
        messagebox.showinfo("Detection Summary", 
                          f"Session Detections: {phone_detection_count}\n"
                          f"Total All-Time Detections: {total_detections}\n"
                          f"Total Runtime: {format_time(total_runtime)}")
    
    phone_detection_count = 0  # Reset session counter after showing the message

# Create buttons after all functions are defined
start_btn = ttk.Button(button_frame, text="Start Detection", command=start_detection)
start_btn.pack(side="left", padx=5, expand=True, fill="x")

stop_btn = ttk.Button(button_frame, text="Stop Detection", command=stop_detection)
stop_btn.pack(side="right", padx=5, expand=True, fill="x")

# Start Tkinter event loop
root.mainloop()