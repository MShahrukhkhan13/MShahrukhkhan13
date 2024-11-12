import cv2
from ultralytics import YOLO
from http.server import BaseHTTPRequestHandler, HTTPServer
import numpy as np
import torch
from collections import deque
import threading
import math

# Check if GPU is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

# Load the YOLOv8 pose model
pose_model = YOLO("Models/yolov8x-pose.pt").to(device)  # Load the pose detection model

# RTSP URL of the network camera
rtsp_url = ''  # Update with your camera's RTSP URL

# Open the RTSP stream
cap = cv2.VideoCapture(rtsp_url)
if not cap.isOpened():
    print(f"Error: Unable to open RTSP stream {rtsp_url}")
    exit()

# Define the buffer size
buffer_size = 15
frame_buffer = deque(maxlen=buffer_size)

# Thresholds for posture classification
threshold_standing = 70  # Example value for standing posture
threshold_sitting = 30    # Example value for sitting posture
walking_threshold = 100   # Example value for walking

# Function to calculate the distance between two points
def distance(pointA, pointB):
    return math.sqrt((pointA[0] - pointB[0]) ** 2 + (pointA[1] - pointB[1]) ** 2)

# Function to calculate the angle between three points
def angle(pointA, pointB, pointC):
    vectorAB = [pointA[0] - pointB[0], pointA[1] - pointB[1]]
    vectorCB = [pointC[0] - pointB[0], pointC[1] - pointB[1]]

    dot_product = vectorAB[0] * vectorCB[0] + vectorAB[1] * vectorCB[1]
    magnitudeAB = math.sqrt(vectorAB[0] ** 2 + vectorAB[1] ** 2)
    magnitudeCB = math.sqrt(vectorCB[0] ** 2 + vectorCB[1] ** 2)

    cos_value = dot_product / (magnitudeAB * magnitudeCB + 1e-6)
    cos_value = max(-1.0, min(1.0, cos_value))  # Clamp the value
    angle_rad = math.acos(cos_value)

    angle_deg = math.degrees(angle_rad) 
    return angle_deg

# Function to classify posture based on keypoints
def classify_posture(keypoints):
    # Check if keypoints is None or if it's a tensor with no elements
    if keypoints is None or (isinstance(keypoints, torch.Tensor) and keypoints.numel() == 0):
        return None  # or handle the error as needed

    # Extract points for easier access
    nose = keypoints[0]
    left_eye, right_eye = keypoints[1], keypoints[2]
    left_ear, right_ear = keypoints[3], keypoints[4]
    left_shoulder, right_shoulder = keypoints[5], keypoints[6]
    left_elbow, right_elbow = keypoints[7], keypoints[8]
    left_wrist, right_wrist = keypoints[9], keypoints[10]
    left_hip, right_hip = keypoints[11], keypoints[12]
    left_knee, right_knee = keypoints[13], keypoints[14]
    left_ankle, right_ankle = keypoints[15], keypoints[16]

    # Calculate distances and angles for posture analysis
    hip_to_ankle_distance = distance(left_hip, left_ankle)
    knee_angle = angle(left_hip, left_knee, left_ankle)
    shoulder_alignment = abs(left_shoulder[1] - right_shoulder[1])
    hip_knee_distance = distance(left_hip, left_knee)

    # Check for standing posture
    if hip_to_ankle_distance > threshold_standing and knee_angle > 150:
        return "Standing"
    
    # Check for sitting posture
    if hip_knee_distance < threshold_sitting and knee_angle < 120:
        return "Sitting"
    
    # Check for walking posture
    if knee_angle > 150 and (left_hip[0] - right_hip[0]) > walking_threshold:
        return "Walking"
    
    return "Sitting"

# Function to read frames and add them to the buffer
def read_frames():
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame_buffer.append(frame)
        else:
            print("Error: Unable to read frame from RTSP stream")
            break

# Start a thread to read frames
read_thread = threading.Thread(target=read_frames)
read_thread.start()

class VideoStreamHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=--jpgboundary')
        self.end_headers()

        while True:
            if frame_buffer:
                frame = frame_buffer.popleft()

                # Preprocess the frame
                frame = cv2.resize(frame, (1280, 960))  # Resize to 1280x960

                # Run YOLOv8 pose model on the frame
                pose_results = pose_model(frame, conf=0.3)

                # Extract boxes and keypoints
                boxes = pose_results[0].boxes.xyxy
                keypoints = pose_results[0].keypoints.xy

                # Check if there are any detected boxes
                if len(boxes) > 0:
                    for i, box in enumerate(boxes):
                        x1, y1, x2, y2 = map(int, box)

                        # Get keypoints for the detected person
                        if i < len(keypoints):  # Ensure we don't go out of bounds
                            keypoints_i = keypoints[i]  # Extract keypoints for current person
                            activity = classify_posture(keypoints_i)  # Classify posture

                            # Set color based on posture
                            if activity == "Sitting":
                                color = (0, 255, 0)  # Green for Sitting
                            elif activity == "Standing":
                                color = (255, 0, 0)  # Blue for Standing
                            else:
                                color = (0, 255, 0)  # Default color (Red) for other postures

                            # Draw bounding box
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                            # Display activity for each detected person
                            cv2.putText(frame, f"{activity}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                # Encode the frame in JPEG format
                ret, jpeg = cv2.imencode('.jpg', frame)
                if not ret:
                    print("Error: Unable to encode frame as JPEG")
                    break

                # Send the frame over HTTP
                self.wfile.write(b'--jpgboundary\r\n')
                self.send_header('Content-type', 'image/jpeg')
                self.send_header('Content-length', str(len(jpeg.tobytes())))
                self.end_headers()
                self.wfile.write(jpeg.tobytes())

# Use 'localhost' for the IP address
httpd = HTTPServer(('localhost', 8095), VideoStreamHandler)
print("Streaming video on http://localhost:8095")
httpd.serve_forever()

# Release the video capture object when done
cap.release()