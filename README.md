Driver Drowsiness Detection using Eye Aspect Ratio (EAR)
Overview
This project implements a real-time driver drowsiness detection system using a webcam. It leverages MediaPipe's Face Mesh and the Eye Aspect Ratio (EAR) method to detect prolonged eye closure, which is an indicator of drowsiness. When drowsiness is detected, the system displays a warning message and plays an audible alert.

Features
Real-time face and eye tracking using MediaPipe Face Mesh

Calculation of EAR to detect eye closure

Visual feedback with eye landmark points drawn on the frame

Alert mechanism: visual warning + beep sound using winsound

Customizable EAR and alert thresholds

How It Works
The system captures video from the webcam using OpenCV.

It tracks specific landmarks of the left and right eye.

The Eye Aspect Ratio (EAR) is calculated for both eyes.

If the EAR falls below a defined threshold for a set number of consecutive frames, an alert is triggered.

Requirements
Python 3.x

OpenCV

MediaPipe

NumPy

Windows OS (for winsound)

Key Parameters
EAR_THRESHOLD = 0.21: Minimum average EAR considered safe

ALERT_THRESHOLD = 15: Number of consecutive low EAR frames before alerting

Usage
Run the script.

A window will display the webcam feed with landmarks drawn on the eyes.

If drowsiness is detected, a warning message appears and a beep sounds.

Press Q to quit.
