# CSc 8830 - Computer Vision
## Assignment 6: Optical Flow and Structure from Motion

This repository contains the implementation of **Assignment 6** for **CSc 8830: Computer Vision**.

The project includes two major parts:

1. **Optical Flow and Motion Tracking**
2. **Structure from Motion using Multiple Camera Views**

The complete system is implemented in Python and displayed through a Streamlit web application.

---

# GitHub Repository

https://github.com/koushikkourikanti/cv_module6

---

# Project Overview

## Part A - Optical Flow

Two videos are used to demonstrate object motion.

The camera remains approximately stationary while the object moves in different directions.

The application performs:

- Dense Optical Flow
- Motion magnitude calculation
- Motion direction estimation
- Optical-flow visualization
- Motion-vector visualization
- Lucas-Kanade feature tracking
- Two-frame motion tracking
- Forward-backward tracking validation
- Bilinear interpolation
- Tracking statistics

The two videos used in this assignment are longer than 30 seconds.

---

# Optical Flow Theory

Optical flow estimates the apparent motion of pixels between consecutive video frames.

The brightness constancy assumption is:

```text
I(x, y, t) = I(x + dx, y + dy, t + dt)