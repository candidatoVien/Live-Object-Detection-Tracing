🍒 Live Object Detection & Tracing 

This project is an aesthetic, high-performance computer vision web application built using Streamlit and YOLOv8. It is designed to provide real-time object detection and automated frame capturing within a customizable, dark-themed user interface.  


✨ Key Features

🧠 Intelligent Detection

YOLOv8 Engine: Uses the YOLOv8 nano model for high-speed and accurate object detection.  

Restricted Target List: Optimized to identify specific items such as people, cell phones, bottles, cats, and laptops.  

Real-time Counting: Includes an on-screen HUD (Heads-Up Display) to track the quantity of detected objects.  

🛠️ Interactive Tools

🪞 Mirror Mode: Allows users to toggle between standard and inverted views.  

🔔 Smart Alerts: Features a visual logging system that timestamps detections for specific targets.  

💾 Auto-Save & Manual Capture: Automatically archives detections every 10 seconds or allows for manual saves.  

🎨 User Experience

Custom Themes: Includes color pickers to modify sidebar and background colors, defaulting to Maroon/Dark Maroon.  

Live Dashboard: A three-column layout that displays real-time stats, alert history, and a gallery of captured frames.  

🛠️ Tech Stack

Streamlit: The frontend framework for the interactive interface.  


Ultralytics YOLOv8: The AI backbone used for object recognition.  

Streamlit-webrtc: Manages low-latency video streaming through the browser.  

OpenCV & PyAV: Handles image processing and frame conversions.  

Threading: Syncs data between the video processor and the UI using thread-safe locks.  

🚀 Getting Started

1. Installation
Install the necessary Python libraries via terminal:  
Bash
pip install streamlit streamlit-webrtc ultralytics opencv-python-headless av numpy pillow

2. Launch the App
Run the following command to start the Streamlit server: 
Bash

streamlit run app.py

📂 Project Structure
app.py: Contains the main application logic and UI layout.  
saved_frames/: The directory where captured detection images are stored as .jpg files.  
yolov8n.pt: Pre-trained YOLOv8 model weights, downloaded automatically during the first run.  

📝 Activity 03 Guidelines Met
This project was designed to exceed the requirements for Activity 03 by including:  
Proof of Observation: A frame-saving system to collect required object screenshots.  
Technical Enhancements: Custom alert logs and aesthetic UI overrides.  
Robust Tracking: Logic that ensures bounding boxes remain accurate even in mirror mode.  

🍷 Usage Tips
Lighting: Best results are achieved in well-lit environments.  
Alerts: Use the sidebar filters to choose which objects trigger critical alerts.  
Data Management: Use the "Delete ALL Saved Frames" button to clear local storage.  