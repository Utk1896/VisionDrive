#  VisionDrive: Vision-Based ADAS System

##  Project Overview

- Real-time ADAS simulation using OpenCV and YOLOv8 on dashcam video
- Combines classical image processing with deep learning
- Provides lane detection, vehicle detection, safety warnings, and visual feedback

---

##  Core Functionalities

- Lane boundary detection and lane area estimation
- Vehicle detection with bounding boxes and confidence scores
- Region of Interest (ROI) masking for focused analysis
- Vehicle counting inside lane boundaries
- Forward collision / crash warning system
- Lane departure warning
- Unified annotated ADAS output video

---

##  Processing Pipeline (Per Frame)

### 1️ Lane Detection

- Grayscale conversion and Gaussian blur
- Canny edge detection
- ROI masking to isolate road region
- Hough Line Transform to detect lane lines
- Lane center and lane polygon estimation

**Output:** Yellow lane overlay and lane polygon

---

### 2️ Safety Zone & Collision Warning

- Defined trapezoidal safety region ahead of vehicle
- Motion detection inside region using background subtraction (MOG2)
- Contour area thresholding to detect obstacles
- Crash alert triggered when risk threshold exceeded

**Output:** Red safety zone, warning text, frame tint during alert

---

### 3️ Vehicle Detection, Tracking & Counting

- YOLOv8 used for detecting cars, trucks, buses, motorcycles
- Tracking enabled with unique IDs
- Centroid calculation for each vehicle
- Vehicles counted only when centroid lies inside lane polygon
- Double counting avoided using `counted_ids` set

**Output:**

- Bounding boxes with Track ID and confidence score
- Green boxes (inside lane), Red boxes (outside lane)
- Live vehicle count display

---

##  Key Design Features

- ROI-based filtering to reduce false positives
- Controlled alert frequency for safety-critical behavior
- Combination of classical CV (edges, Hough) and YOLO detection
- Real-time frame-by-frame processing and visualization

---

##  Final Output

Single annotated video stream showing:

- Lane overlays
- Vehicle detections
- ROI mask
- Vehicle count
- Collision and lane departure warnings
