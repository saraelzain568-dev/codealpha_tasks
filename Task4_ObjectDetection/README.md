# AI Object Detection and Tracking

## Project Overview

This project is an AI-based Object Detection and Tracking application built with Python, Streamlit, OpenCV, and YOLO.

The application can analyze images and videos, detect objects, draw bounding boxes around detected objects, and track objects in video streams.

## Task Objectives

* Set up image and video input.
* Use a pre-trained YOLO model for object detection.
* Process images and video frames using OpenCV.
* Detect and classify objects automatically.
* Draw bounding boxes and object labels.
* Apply object tracking to video frames.
* Display detection and tracking results through a Streamlit interface.

## Technologies Used

* Python
* Streamlit
* YOLO
* Ultralytics
* OpenCV
* PyTorch

## Features

* Image object detection
* Video object detection
* Object tracking
* Bounding boxes
* Object labels
* Streamlit web interface
* Real-time frame processing
* Pre-trained YOLO model

## Project Structure

```text
Task4_ObjectDetection/
│
├── app.py
├── requirements.txt
├── .gitignore
├── README.md
└── screenshots/
    ├── task4_detection_interface.png
    └── task4_detection_result.png
```

## How It Works

The user uploads an image or video through the Streamlit interface.

For images, the YOLO model analyzes the image and detects available objects. The application then displays the detected objects with bounding boxes and labels.

For videos, the application processes the video frame by frame. YOLO detects objects and the tracking system follows detected objects across consecutive frames.

## How to Run

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

The application will open in the browser.

## Model

The project uses a pre-trained YOLO model from Ultralytics for object detection and tracking.

The model file is downloaded automatically when required and is not included in the GitHub repository.

## Screenshots

The `screenshots` folder contains examples of the application's interface and object detection results.

### Application Interface

The interface allows the user to upload an image or video for AI-based object detection and tracking.

### Detection Result

The detection result shows the objects detected by YOLO with bounding boxes and labels.

## Result

The application successfully demonstrates AI-based object detection and tracking using YOLO, OpenCV, and Streamlit.

## Author

CodeAlpha Internship – Task 4
