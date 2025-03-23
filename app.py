import streamlit as st
from src.frame_processor import process_frame
import time
import cv2
import tempfile
import os
import plotly.graph_objects as go

st.set_page_config(page_title="AttenSense", layout="centered")

# Custom CSS
st.markdown(
    """ <style>
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
        
        /* css for focus status */
        .focus-text {
            text-align: center;
            font-size: 50px;
            font-weight: bold;
            margin-top: 10px;
        }
        .focused {
            color: green;
        }
        .not-focused {
            color: red;
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

#input phase

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
            st.session_state.analysis_phase = "analyzing"
            st.session_state.focus_log = []
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

    start_time = time.time()
    stframe = st.empty()
    focus_status_placeholder = st.empty()

    if st.session_state.input_method == "Webcam":
        st.write("Live webcam stream activated. Stay focused!")
        cap = cv2.VideoCapture(0)  # 0 for default webcam
        # stframe = st.empty()
        # focus_status_placeholder = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture image from webcam.")
                break
            
            
            

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            

            # Process frame and get focus status
            frame, is_focused = process_frame(frame)

            current_time = time.time() - start_time
            st.session_state.focus_log.append((current_time, is_focused))


            stframe.image(frame, channels="RGB")

            # Display focus status
            status_text = "Focused" if is_focused else "Not Focused"
            color_class = "focused" if is_focused else "not-focused"
            focus_status_placeholder.markdown(
                f'<p class="focus-text {color_class}" style="font-size: 50px;">{status_text}</p>', 
                unsafe_allow_html=True
            )
            if st.session_state.analysis_phase != "analyzing":
                break


        cap.release()

    elif st.session_state.uploaded_video_path is not None:
        st.write("Processing your video... Sit tight!")
        cap = cv2.VideoCapture(st.session_state.uploaded_video_path)
        # stframe = st.empty()
        # focus_status_placeholder = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.warning("Video processing complete.")
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process frame and get focus status
            frame, is_focused = process_frame(frame)

            #focus log
            current_time = time.time() - start_time
            st.session_state.focus_log.append((current_time, is_focused))


            stframe.image(frame, channels="RGB")
            status_text = "Focused" if is_focused else "Not Focused"
            color_class = "focused" if is_focused else "not-focused"

            focus_status_placeholder.markdown(
                f'<p class="focus-text {color_class}" style="font-size: 50px;">{status_text}</p>', 
                unsafe_allow_html=True
            )


            time.sleep(0.03)

        cap.release()
        st.session_state.analysis_phase = "analysis_complete"
        st.rerun()

#report phase
if st.session_state.analysis_phase == "analysis_complete":
    focus_log = st.session_state.focus_log
    if focus_log:
        frame_duration = 0.03  # Approx 30 fps
        focused_duration = sum(1 for _, f in focus_log if f) * frame_duration
        total_duration = len(focus_log) * frame_duration
        unfocused_duration = total_duration - focused_duration
        focus_percentage = (focused_duration / total_duration) * 100 if total_duration > 0 else 0

        st.subheader("📊 Focus Report")
        st.write(f"**Total Duration:** {total_duration:.2f} seconds")
        st.write(f"**Focused Duration:** {focused_duration:.2f} seconds")
        st.write(f"**Unfocused Duration:** {unfocused_duration:.2f} seconds")
        st.write(f"**Focus Percentage:** {focus_percentage:.2f}%")

        timestamps = [t for t, _ in focus_log]
        focus_values = [1 if f else 0 for _, f in focus_log]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=timestamps, y=focus_values, mode='lines+markers',
                                 name='Focus Status',
                                 line=dict(color='green'),
                                 marker=dict(color=['green' if f else 'red' for f in focus_values])))

        fig.update_layout(title='Focus Over Time',
                          xaxis_title='Time (s)',
                          yaxis_title='Focus (1=Focused, 0=Not Focused)',
                          yaxis=dict(tickmode='array', tickvals=[0, 1],
                                     ticktext=["Not Focused", "Focused"]),
                          height=400)

        st.plotly_chart(fig, use_container_width=True)

    if st.button("Return to Home"):
        st.session_state.analysis_phase = "idle"
        st.session_state.focus_log = []
        st.session_state.uploaded_video_path = None
        st.rerun()