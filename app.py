import pandas as pd
from io import StringIO
import streamlit as st
from src.frame_processor import process_frame
import time
import cv2
import tempfile
import os
import altair as alt

def group_focus_log(focus_log):
    if not focus_log:
        return []
    
    grouped_log = []
    start_time, state = focus_log[0]  # Initialize with first entry

    for i in range(1, len(focus_log)):
        current_time, current_state = focus_log[i]

        # If state changes, store the previous segment
        if current_state != state:
            grouped_log.append((start_time, focus_log[i - 1][0], state))  # (start, end, state)
            start_time = current_time  # Update start time for the new state
            state = current_state  # Update state

    # Append last segment
    grouped_log.append((start_time, focus_log[-1][0], state))
    
    return grouped_log

def create_data_frame(focus_log):
    grouped_focus_log = group_focus_log(focus_log)
    df = pd.DataFrame(grouped_focus_log, columns = ["Start", "End", "Focused"])
    df["Duration"] = df["End"] - df["Start"]
    df["Focus State"] = df["Focused"].map({True: "Focused", False: "Unfocused"})
    df.drop("Focused", axis = 1, inplace = True)

    return df

def create_altair_chart(df):
    chart = alt.Chart(df).mark_bar().encode(
        x="Start:Q",
        x2="End:Q",
        y=alt.Y("Focus State:N", title="Focus State"),
        color="Focus State:N"
    ).properties(title="Focus Over Time")
    return chart

st.set_page_config(page_title = "AttenSense", layout = "centered")

# Custom CSS
st.markdown(
    """
    <style>
    """
    <style>
        /* setting bg colour */
        [data-testid="stAppViewContainer"] {
            background-color: #FDFBEE;
        }

        /* css for title */
        .title {
            text-align: center;
            color: #ff6347;
            font-size: 36px;
            font-weight: bold;
        }
        /* css for button */
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
            color: black;
            transform: scale(1.02); 
        }

        /* Change the border color of the file uploader */
        div[data-testid="stFileUploader"] {
            border: 2px solid #ff6347 !important; 
            border-radius: 10px !important;
            padding: 15px !important;
        }

        /* Change border color of the button inside file uploader */
        div[data-testid="stFileUploader"] > label {
            border: 2px solid #C1CFA1 !important; 
            background-color: #f9f9f9 !important; 
            border-radius: 10px !important;
            padding: 10px !important;
        }

        /* Change the border color of Webcam button */
        div[data-testid="stCameraInput"] {
            border: 2px solid #ff6347 !important;  
            border-radius: 10px !important;
            padding: 10px !important;
        }

        /* Setting border css for upload video section */
        div[data-testid="stFileUploader"] {
            border: 2px dashed #A5B68D !important;
            border-radius: 10px !important;
            padding: 15px !important;
        }

         /* Styling Radio Buttons */
        div[role="radiogroup"] label[data-baseweb="radio"] > div:first-child {
            background-color: #A5B68D !important; 
            border: 2px solid #A5B68D !important; 
        }

         /* Target the Browse files button */
        div[data-testid="stFileUploader"] button {
            background-color: #C1CFA1 !important;  
            color: black !important;              
            font-size: 16px !important;
            padding: 8px 15px !important;
            border: 1px solid #C1CFA1!important;
            transition: all 0.3s ease-in-out;
        }

        /* Change the hover effect */
        div[data-testid="stFileUploader"] button:hover {
            background-color: #A5B68D !important;  
            color: black !important;              
            transform: scale(1.02);               
        }
        /* CSS for Focus / Not Focused */
        .focused {
            color: green;
        }

        /* Change the hover effect */
        div[data-testid="stFileUploader"] button:hover {
            background-color: #A5B68D !important;  
            color: black !important;              
            transform: scale(1.02);               
        }
    </style>
    """,
    unsafe_allow_html=True
)

uploads_path = os.path.join(os.getcwd(), "uploads")
os.makedirs(uploads_path, exist_ok=True)

# Title and Desc
st.markdown('<h1 class="title">AttenSense</h1>', unsafe_allow_html=True)
st.write("An AI-based tool to analyze human attention using computer vision.")

# Initialize session state
if "analysis_phase" not in st.session_state:
    st.session_state.analysis_phase = "idle"
if "input_method" not in st.session_state:
    st.session_state.input_method = None
if "uploaded_video" not in st.session_state:
    st.session_state.uploaded_video = None
if "uploaded_video_path" not in st.session_state:
    st.session_state.uploaded_video_path = None
if "focus_log" not in st.session_state:
    st.session_state.focus_log = []

# Input Phase
if st.session_state.analysis_phase == "idle":
    # Upload Video or Use Webcam
    video_source = st.radio("Choose Your Video Input:", ("Webcam", "Upload Video"))

    uploaded_video = None
    if video_source == "Upload Video":
        uploaded_video = st.file_uploader("Upload a Video File (MP4, AVI, MOV)", type=["mp4", "avi", "mov"])

    if st.button("Begin Attention Analysis"):
        # Set session state for input method
        if video_source == "Upload Video" and uploaded_video is None:
            st.warning("Kindly upload a video to proceed with the analysis.")
        else:
            st.session_state.input_method = video_source
            st.session_state.start_time = time.time()
            st.session_state.analysis_phase = "analyzing"
            if uploaded_video is not None:
                temp_file = tempfile.NamedTemporaryFile(dir=uploads_path, delete=False, suffix=".mp4")
                temp_file.write(uploaded_video.read())
                temp_file.flush()
                st.session_state.uploaded_video_path = temp_file.name
            st.rerun()

# If analysis has started, handle the input type
if st.session_state.analysis_phase == "analyzing":
    if st.button("End Attention Analysis"):
        st.session_state.input_method = None
        st.session_state.analysis_phase = "analysis_complete"
        st.session_state.uploaded_video = None
        st.rerun()

    if st.session_state.input_method == "Webcam":
        st.write("Live webcam stream activated. Stay focused!")
        cap = cv2.VideoCapture(0)  # 0 for default webcam
        stframe = st.empty()
        focus_status_placeholder = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture image from webcam.")
                break
            
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            
            # Process frame and get focus status
            frame, is_focused = process_frame(frame)

            start_time = st.session_state.start_time
            current_time = time.time() - start_time
            st.session_state.focus_log.append((current_time, is_focused))

            stframe.image(frame, channels="RGB")

            # Display focus status
            status_text = "Focused" if is_focused else "Not Focused"
            color_class = "focused" if is_focused else "not-focused"

            focus_status_placeholder.markdown(
                f'<p class="focus-text {color_class}" style="font-size: 36px; font-weight: bold; text-align: center; padding: 10px; border-radius: 10px; background-color: #FDFBEE">{status_text}</p>',
                unsafe_allow_html=True
            )
            time.sleep(0.05)

        cap.release()

    elif st.session_state.uploaded_video_path is not None:
        st.write("Processing your video... Sit tight!")
        cap = cv2.VideoCapture(st.session_state.uploaded_video_path)
        stframe = st.empty()
        focus_status_placeholder = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.warning("Video processing complete.")
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            
            # Process frame and get focus status
            frame, is_focused = process_frame(frame)

            start_time = st.session_state.start_time
            current_time = time.time() - start_time
            st.session_state.focus_log.append((current_time, is_focused))

            stframe.image(frame, channels="RGB")
            status_text = "Focused" if is_focused else "Not Focused"
            color_class = "focused" if is_focused else "not-focused"

            focus_status_placeholder.markdown(
                f'<p class="focus-text {color_class}" style="font-size: 36px; font-weight: bold; text-align: center; padding: 10px; border-radius: 10px; background-color: #FDFBEE">{status_text}</p>',
                unsafe_allow_html=True
            )
            time.sleep(0.05)

        cap.release()

# Report phase
if st.session_state.analysis_phase == "analysis_complete":
    focus_log = st.session_state.focus_log
    if focus_log:
        frame_duration = 0.05  # 20 fps
        total_duration = len(focus_log) * frame_duration
        focused_duration = sum(1 for _, f in focus_log if f) * frame_duration
        unfocused_duration = total_duration - focused_duration
        focus_percentage = (focused_duration / total_duration) * 100 if total_duration > 0 else 0

        st.subheader("📊 Focus Report")
        st.write(f"**Total Duration:** {total_duration:.2f} seconds")
        st.write(f"**Focused Duration:** {focused_duration:.2f} seconds")
        st.write(f"**Unfocused Duration:** {unfocused_duration:.2f} seconds")
        st.write(f"**Focus Percentage:** {focus_percentage:.2f}%")

        focus_df = create_data_frame(focus_log)
        chart = create_altair_chart(focus_df)

        st.altair_chart(chart, use_container_width=True)
        
        # Convert to CSV
        csv_data = focus_df.to_csv(index = False)

        # Provide a download button
        st.download_button(
            label = "📥 Download Focus Report",
            data = csv_data,
            file_name = "focus_report.csv",
            mime = "text/csv"
        )
    
    if st.button("Return to Home"):
        st.session_state.analysis_phase = "idle"
        st.session_state.focus_log = []
        st.session_state.uploaded_video_path = None
        st.session_state.focus_log = []
        st.session_state.uploaded_video_path = None
        st.rerun()