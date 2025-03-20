import streamlit as st
import av
import cv2
import time
import tempfile
import os
import pandas as pd
import matplotlib.pyplot as plt
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# Initialize session state variables
if "analysis_phase" not in st.session_state:
    st.session_state.analysis_phase = "idle"
if "input_method" not in st.session_state:
    st.session_state.input_method = None
if "uploaded_video_path" not in st.session_state:
    st.session_state.uploaded_video_path = None
if "focus_log" not in st.session_state:
    st.session_state.focus_log = []

# Define storage path
uploads_path = tempfile.gettempdir()

# Custom CSS
st.markdown(
    """
    <style>
        [data-testid="stAppViewContainer"] {
            background-color: #FDFBEE;
        }
        .title {
            text-align: center;
            color: #ff6347;
            font-size: 36px;
            font-weight: bold;
        }
        div.stButton > button {
            background-color: #C1CFA1;
            color: black;
            font-size: 20px;
            padding: 10px 20px;
            border-radius: 10px;
            border: none;
            transition: 0.3s;
        }
        div.stButton > button:hover {
            background-color: #A5B68D;
            transform: scale(1.02);
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Title and description
st.markdown('<h1 class="title">AttenSense</h1>', unsafe_allow_html=True)
st.write("An AI-based tool to analyze human attention using computer vision.")

# Input selection
video_source = st.radio("Select Input Source:", ("Webcam", "Upload Video"))
st.session_state.input_method = video_source

uploaded_file = None
if video_source == "Upload Video":
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

# Define face detection function
def process_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    is_focused = len(faces) > 0
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return frame, is_focused

# Handle analysis phases
if st.session_state.analysis_phase == "idle":
    if st.button("Begin Attention Analysis"):
        if video_source == "Upload Video" and uploaded_file is None:
            st.warning("Please upload a video.")
        else:
            st.session_state.analysis_phase = "analyzing"
            st.session_state.focus_log = []

            if uploaded_file is not None:
                temp_file = tempfile.NamedTemporaryFile(dir=uploads_path, delete=False, suffix=".mp4")
                temp_file.write(uploaded_file.read())
                temp_file.flush()
                st.session_state.uploaded_video_path = temp_file.name
            
            st.rerun()

if st.session_state.analysis_phase == "analyzing":
    if st.button("End Attention Analysis"):
        st.session_state.analysis_phase = "analysis_complete"
        st.rerun()

    if st.session_state.input_method == "Webcam":
        st.write("Live webcam stream activated. Stay focused!")
        cap = cv2.VideoCapture(0)
        stframe = st.empty()
        focus_status_placeholder = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("Webcam error.")
                break
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame, is_focused = process_frame(frame)
            stframe.image(frame, channels="RGB")

            status_text = "Focused" if is_focused else "Not Focused"
            focus_status_placeholder.markdown(f'<p style="font-size: 50px;">{status_text}</p>', unsafe_allow_html=True)
            st.session_state.focus_log.append((time.time(), is_focused))

        cap.release()

    elif st.session_state.uploaded_video_path:
        st.write("Processing your video...")
        cap = cv2.VideoCapture(st.session_state.uploaded_video_path)
        stframe = st.empty()
        focus_status_placeholder = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.warning("Video processing complete.")
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame, is_focused = process_frame(frame)
            stframe.image(frame, channels="RGB")
            focus_status_placeholder.markdown(f'<p style="font-size: 50px;">{"Focused" if is_focused else "Not Focused"}</p>', unsafe_allow_html=True)
            st.session_state.focus_log.append((time.time(), is_focused))
            time.sleep(0.03)

        cap.release()

if st.session_state.analysis_phase == "analysis_complete":
    st.write("### Focus Session Summary")
    df = pd.DataFrame(st.session_state.focus_log, columns=["Timestamp", "Focused"])
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="s")
    df["Focused"] = df["Focused"].astype(int)
    focus_percentage = df["Focused"].mean() * 100

    st.write(f"**Total Time Logged:** {len(df)} seconds")
    st.write(f"**Percentage of Time Focused:** {focus_percentage:.2f}%")

    fig, ax = plt.subplots()
    ax.plot(df["Timestamp"], df["Focused"], marker="o", linestyle="-", linewidth=1, color="blue", label="Focus Level")
    ax.set_xlabel("Time")
    ax.set_ylabel("Focus (0 = Not Focused, 1 = Focused)")
    ax.set_title("Focus Over Time")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Focus Log", csv, "focus_log.csv", "text/csv", key='download-csv')

    if st.button("Return to Home"):
        st.session_state.analysis_phase = "idle"
        st.rerun()
