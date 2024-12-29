
import tensorflow as tf
import cv2
model_path = r"E:\Projects\tensor\ssd_mobilenet_v1_fpn_640x640_coco17_tpu-8\ssd_mobilenet_v1_fpn_640x640_coco17_tpu-8\saved_model"
detection_model = tf.saved_model.load(model_path)

# Function to detect objects
def detect_objects(image):
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis, ...]
    detections = detection_model(input_tensor)
    return detections

# Load webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect objects
    detections = detect_objects(rgb_frame)

    # Process detections (adjust indices based on model output)
    for i in range(int(detections['num_detections'][0])):
        class_id = int(detections['detection_classes'][0][i])
        score = float(detections['detection_scores'][0][i])
        bbox = detections['detection_boxes'][0][i]

        if score > 0.5:  # Confidence threshold
            h, w, _ = frame.shape
            x1, y1, x2, y2 = int(bbox[1] * w), int(bbox[0] * h), int(bbox[3] * w), int(bbox[2] * h)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"Class {class_id}: {score:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show the output
    cv2.imshow("Webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



