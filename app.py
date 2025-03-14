import streamlit as st
import time
import cv2
import tempfile
import os

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

uploads_path = os.path.join(os.getcwd(),"uploads")
os.makedirs(uploads_path, exist_ok = True)

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

if st.session_state.analysis_phase == "idle" :
    # Upload Video or Use Webcam
    video_source = st.radio("Choose Your Video Input:", ("Webcam", "Upload Video"))

    uploaded_video = None
    if video_source == "Upload Video":
        uploaded_video = st.file_uploader("Upload a Video File (MP4, AVI, MOV)", type=["mp4", "avi", "mov"])

    if st.button("Begin Attention Analysis") :
        # Set session state for input method
        if video_source == "Upload Video" and uploaded_video == None :
            st.warning("Kindly upload a video to proceed with the analysis.")
        else :
            st.session_state.input_method = video_source
            st.session_state.analysis_phase = "analyzing"
            if uploaded_video != None : 
                temp_file = tempfile.NamedTemporaryFile(dir = uploads_path, delete = False, suffix = ".mp4")
                temp_file.write(uploaded_video.read())
                temp_file.flush()
                st.session_state.uploaded_video_path = temp_file.name
            st.rerun()
        
            
# If analysis has started, handle the input type
if st.session_state.analysis_phase == "analyzing" :
    if st.button("End Attention Analysis") :
        st.session_state.input_method = None
        st.session_state.analysis_phase = "analysis_complete"
        st.session_state.uploaded_video = None
        st.rerun()

    if st.session_state.input_method == "Webcam":
        st.write("Live webcam stream activated. Stay focused!")
        # OpenCV Webcam Streaming

        cap = cv2.VideoCapture(0)  # 0 for default webcam
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture image from webcam.")
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB for Streamlit
            stframe.image(frame, channels="RGB")

        cap.release()

    elif st.session_state.uploaded_video_path is not None:
        st.write("Processing your video... Sit tight!")
        # Process video file
        cap = cv2.VideoCapture(st.session_state.uploaded_video_path)
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.warning("Video processing complete.")
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB for Streamlit
            stframe.image(frame, channels="RGB")
            time.sleep(0.03)

        cap.release()

if st.session_state.analysis_phase == "analysis_complete" :
    if st.button("Return to Home") :
        st.session_state.analysis_phase = "idle"
        st.rerun()