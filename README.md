# RTSP Pose Detection and Posture Classification with YOLOv8
This repository provides a Python script that captures video frames from an RTSP stream, detects human poses using the YOLOv8 model, and classifies the detected poses into activities such as Standing and Sitting. The script includes preliminary code for Walking posture classification, though this feature is experimental and may require further adjustments to function accurately. The processed frames are streamed over HTTP, enabling easy access through a web browser

## Features
- **Pose Detection** : Detects human poses in real-time using YOLOv8.
- **Posture Classification** : Classifies each detected pose into activities (Standing, Sitting, Walking - experimental).
- **Stream Output** : Serves the processed video frames as an HTTP stream, viewable in any browser.
- **GPU Support** : Leverages CUDA if available, for faster processing.

## Prerequisites
- Python 3.7+
- NVIDIA GPU with CUDA support (optional for faster processing)
- An RTSP-enabled camera

## Installation
- **Clone the repository:**
   > git clone https://github.com/MShahrukhkhan13/PoseDetection.git

## Configuration
Update the rtsp_url variable in the script with the RTSP URL of your camera:
> rtsp_url = 'rtsp://your-camera-url'

## Posture Classification Thresholds
Customize thresholds based on your requirements:
- threshold_standing: Distance threshold for standing posture.
- threshold_sitting: Distance threshold for sitting posture.
- walking_threshold: Distance threshold for walking posture (requires testing and adjustments).
  
These values define criteria for distinguishing between different activities.

## Usage
Run the script:

> python PoseDetection.py

## Script Overview
### Key Functions
- **distance** : Calculates Euclidean distance between two points.
- **angle** : Calculates the angle between three points, which helps classify the posture.
- **classify posture** : Classifies each detected pose as Standing, Sitting, or Walking (experimental) based on keypoints and predefined thresholds.
- **VideoStreamHandler** : Handles the HTTP GET request and streams processed frames with bounding boxes and posture labels.

### Main Logic
- **Initialize YOLOv8 Model** : Loads the pose detection model on GPU (if available) or CPU.
- **Open RTSP Stream** : Captures frames from the specified RTSP URL.
- **Posture Classification** : Each frame is analyzed for posture, using keypoints to calculate angles and distances.
- **HTTP Stream** - Streams frames over HTTP in MJPEG format, viewable in a browser.

## Notes on Walking Posture
The Walking classification feature is currently experimental. The threshold values and posture logic may require tuning depending on the camera angle, lighting, and distance from the camera. Testing and adjustments are recommended if accurate walking classification is needed.

## Example Output
The output consists of a video stream with bounding boxes around detected individuals and labels indicating their posture (e.g., Standing, Sitting, Walking - experimental).


