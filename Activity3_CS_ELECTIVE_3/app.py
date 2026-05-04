import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from ultralytics import YOLO
import av
import cv2
from collections import defaultdict
from datetime import datetime
import os
import time
import numpy as np
import threading
import shutil
from PIL import Image

# Directory setup for frame storage[cite: 2]
SAVED_FRAMES_DIR = "saved_frames"
if not os.path.exists(SAVED_FRAMES_DIR):
    os.makedirs(SAVED_FRAMES_DIR)

# Initialize session state with Maroon/Dark Maroon theme[cite: 2]
if 'sidebar_bg' not in st.session_state:
    st.session_state.sidebar_bg = "#800000" 
if 'page_bg' not in st.session_state:
    st.session_state.page_bg = "#4D0000"
if 'camera_active' not in st.session_state:
    st.session_state.camera_active = False

class SharedData:
    def __init__(self):
        self.object_counts = defaultdict(int)
        self.detection_log = []
        self.last_alert_time = 0
        self.last_auto_save_time = 0
        self.mirror_view_enabled = True
        self.show_counting = True
        self.enable_alerts = True
        # Restricted target list[cite: 2]
        self.allowed_targets = ["person", "cell phone", "bottle", "cat", "fork", "spoon", "scissors", "laptop", "book"]
        self.alert_objects = ["person"]
        self.auto_save = False
        self.save_request = False
        self.lock = threading.Lock()
    
    def update_data(self, counts, detections):
        with self.lock:
            self.object_counts = counts
            now = time.time()
            if self.enable_alerts and (now - self.last_alert_time) > 2:
                for det in detections:
                    if det in self.alert_objects:
                        self.detection_log.append({
                            'object': det,
                            'timestamp': datetime.now().strftime("%H:%M:%S")
                        })
                        self.last_alert_time = now
                        if len(self.detection_log) > 10: self.detection_log.pop(0)
                        break

    def get_state(self):
        with self.lock:
            return {
                "counts": self.object_counts.copy(),
                "alerts": self.detection_log.copy(),
                "mirror": self.mirror_view_enabled,
                "show_counting": self.show_counting,
                "save_req": self.save_request,
                "auto_save": self.auto_save
            }

if 'shared' not in st.session_state:
    st.session_state.shared = SharedData()
shared = st.session_state.shared

@st.cache_resource
def load_yolo():
    return YOLO("yolov8n.pt")

model = load_yolo()

# Specific Color Mapping for labels[cite: 2]
OBJ_COLORS = {
    'person': (180, 105, 255), 'bottle': (0, 0, 255), 'fork': (128, 0, 128),
    'cell phone': (128, 0, 128), 'spoon': (0, 255, 0), 'scissors': (0, 0, 128),
    'cat': (255, 192, 203), 'laptop': (255, 255, 0), 'book': (0, 165, 255),
    'default': (255, 255, 255)
}

class VideoProcessor:
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        h, w = img.shape[:2]
        state = shared.get_state()
        results = model(img, verbose=False, conf=0.5)
        curr_counts, curr_dets = defaultdict(int), []
        
        raw_detections = []
        if results and results[0].boxes:
            for box in results[0].boxes:
                name = model.names[int(box.cls[0])]
                curr_counts[name] += 1
                curr_dets.append(name)
                raw_detections.append({'box': box.xyxy[0].tolist(), 'name': name})

        shared.update_data(curr_counts, curr_dets)
        if state["mirror"]: img = cv2.flip(img, 1)

        for det in raw_detections:
            x1, y1, x2, y2 = map(int, det['box'])
            name = det['name']
            color = OBJ_COLORS.get(name, OBJ_COLORS['default'])
            if state["mirror"]:
                new_x1 = w - x2
                new_x2 = w - x1
                x1, x2 = new_x1, new_x2
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
            cv2.putText(img, name.upper(), (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        if state["show_counting"]:
            y = 30
            for obj, count in state["counts"].items():
                if count > 0:
                    cv2.putText(img, f"{obj}: {count}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                    y += 25
        
        if state["mirror"]:
            cv2.putText(img, "MIRROR MODE", (w-160, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 128), 2)

        # Logic for saving frames[cite: 2]
        if state["save_req"] or (state["auto_save"] and time.time() - shared.last_auto_save_time > 10):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(f"{SAVED_FRAMES_DIR}/frame_{timestamp}.jpg", img)
            with shared.lock:
                shared.save_request = False
                shared.last_auto_save_time = time.time()

        return av.VideoFrame.from_ndarray(img, format="bgr24")

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Quicksand:wght@500;700&family=Dancing+Script:wght@700&display=swap');
    .stApp {{ background-color: {st.session_state.page_bg} !important; font-family: 'Quicksand'; }}
    [data-testid="stSidebar"] {{ background-color: {st.session_state.sidebar_bg} !important; }}
    .aesthetic-header {{ font-family: 'Dancing Script', cursive; font-size: 3rem; text-align: center; color: white !important; margin-bottom: 5px; }}
    .site-desc {{ font-size: 0.9rem; color: rgba(255, 255, 255, 0.8) !important; margin-bottom: 20px; font-style: italic; }}
    .brown-alert {{ background-color: #5D4037; color: white !important; padding: 10px; border-radius: 8px; margin: 5px 0; border-left: 5px solid #3E2723; font-size: 0.85rem; }}
    .frame-card {{ border: 2px solid #800000; border-radius: 10px; margin-bottom: 10px; overflow: hidden; }}
    h1, h2, h3, h4, p, span, label {{ color: white !important; font-family: 'Quicksand'; }}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("<h2 style='text-align: justified;'>🍷Settings🍷</h2>", unsafe_allow_html=True)
    st.markdown("<div class='site-desc'>This project focuses on developing an interactive web application that integrates real-time video streaming with a YOLOv8 neural network to identify and trace everyday objects. Learners explore the fundamentals of computer vision by deploying AI models in a live environment, gaining hands-on experience with frame-by-frame image processing and object tracking. To achieve a maximum grade, the activity includes technical enhancements such as real-time object counting, specific target alerts, and an automated system for saving detected frames.</div>", unsafe_allow_html=True)
    
    st.markdown("### 📷 Camera Settings")
    shared.mirror_view_enabled = st.checkbox("🪞 Mirror View (Inverted)", value=True)
    st.markdown("### 🗺️ Quality & Resolution")
    st.selectbox("Select Resolution", ["Low (480p)", "Medium (720p)", "High (1080p)"])
    shared.auto_save = st.checkbox("💾 Auto-save every 10 seconds")
    st.markdown("### 📊 Object Counting")
    shared.show_counting = st.checkbox("🧮 Show Object Counting", value=True)
    st.markdown("### 🛎️ Alert System")
    shared.enable_alerts = st.checkbox("🔔 Enable Alerts", value=True)
    shared.alert_objects = st.multiselect("🎯 Alert for these objects", shared.allowed_targets, default=["person"])
    
    if st.button("🔄 Reset All Counters"):
        with shared.lock:
            shared.object_counts.clear()
            shared.detection_log.clear()

    st.markdown("---")
    st.markdown("### 🎨 Theme Customization")
    st.session_state.sidebar_bg = st.color_picker("Sidebar Background Color", st.session_state.sidebar_bg)
    st.session_state.page_bg = st.color_picker("Page Background Color", st.session_state.page_bg)
    
    if st.button("🗑️ Delete ALL Saved Frames"):
        shutil.rmtree(SAVED_FRAMES_DIR)
        os.makedirs(SAVED_FRAMES_DIR)
        st.success("All frames deleted!")

st.markdown("<h2 class='aesthetic-header'> 🍒Live Object Detection & Tracing🍒</h2>", unsafe_allow_html=True)
st.markdown("<h3 class='aesthetic-header'> 🍷Point your camera at objects to identify them in real-time with AI magic🍷</h3>", unsafe_allow_html=True)

if st.session_state.camera_active:
    webrtc_streamer(key="yolo", video_processor_factory=VideoProcessor, media_stream_constraints={"video": True, "audio": False})
    if st.button("⏹️ Stop Camera"):
        st.session_state.camera_active = False
        st.rerun()
else:
    if st.button("📷 Start Camera", type="primary"):
        st.session_state.camera_active = True
        st.rerun()

@st.fragment(run_every=2)
def show_dashboard():
    state = shared.get_state()
    # Three-column layout: Counts | Alerts | Saved Frames[cite: 2]
    col1, col2, col3 = st.columns([1, 1.5, 1.5])
    
    with col1:
        st.markdown("### 📊 Counts")
        for k, v in state["counts"].items():
            if v > 0: st.write(f"**{k}:** {v}")
            
    with col2:
        st.markdown("### 🚨 Alerts")
        for a in reversed(state["alerts"]):
            st.markdown(f"<div class='brown-alert'>🚨 <b>target {a['object']}</b> detected at {a['timestamp']}</div>", unsafe_allow_html=True)
            
    with col3:
        st.markdown("### 🖼️ Saved Frames")
        files = sorted([f for f in os.listdir(SAVED_FRAMES_DIR) if f.endswith('.jpg')], reverse=True)
        if files:
            for img_file in files[:5]: # Show latest 5 frames[cite: 2]
                img_path = os.path.join(SAVED_FRAMES_DIR, img_file)
                st.image(img_path, caption=f"Captured: {img_file.split('_')[2].replace('.jpg','')}", use_container_width=True)
        else:
            st.write("No frames saved yet.")

show_dashboard()