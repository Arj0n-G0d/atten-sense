import streamlit as st
import av
import cv2
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# custom CSS
st.markdown(
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
    </style>
    """,
    unsafe_allow_html=True
)

# Title and Desc
st.markdown('<h1 class="title">AttenSense</h1>', unsafe_allow_html=True)
st.write("An AI-based tool to analyze human attention using computer vision.")

# Initialize session state
if "analysis_started" not in st.session_state:
    st.session_state.analysis_started = False
if "input_method" not in st.session_state:
    st.session_state.input_method = None

# Upload Video or Use Webcam
video_source = st.radio("Select Input Source:", ("Webcam", "Upload Video"))

# Set session state for input method
st.session_state.input_method = video_source

uploaded_file = None 
if video_source == "Upload Video":
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

# Webcam Streaming Setup
class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        # Future: Add AI Processing Here (e.g., Face Detection)       
        return av.VideoFrame.from_ndarray(img, format="bgr24")

if st.button("Start Analysis"):
    if st.session_state.input_method == "Webcam":
        st.session_state.analysis_started = True
    elif uploaded_file is not None:
        st.session_state.analysis_started = True
    else:
        st.warning("Please upload a video before starting analysis.")
        
# If analysis has started, handle the input type
if st.session_state.analysis_started:
    if st.session_state.input_method == "Webcam":
        st.write("Starting live webcam stream...")
        # OpenCV Webcam Streaming
        webrtc_streamer(key="focus-analyser-stream", video_processor_factory=VideoProcessor)
        # cap = cv2.VideoCapture(0)  # 0 for default webcam
        # stframe = st.empty()

        # while cap.isOpened():
        #     ret, frame = cap.read()
        #     if not ret:
        #         st.error("Failed to capture image from webcam.")
        #         break

        #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB for Streamlit
        #     stframe.image(frame, channels="RGB")

        # cap.release()

    elif uploaded_file is not None:
        st.write("Analyzing uploaded video...")
        # Process video file
        video_bytes = uploaded_file.read()
        st.video(video_bytes)  # Display video
