import numpy as np
import cv2
import dlib
import pandas as pd
import time
from scipy.spatial import distance

# Initialize detector and predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r"C:\Users\anupm\OneDrive\Desktop\AttenSense\src\shape_predictor_68_face_landmarks.dat")

# Eye landmark indices
LEFT_EYE = list(range(42, 48))
RIGHT_EYE = list(range(36, 42))

# Store focus logs
focus_logs = []  # Stores (timestamp, focus_state)
focus_start_time = None  # Track when focus started
total_focus_time = 0  # Track total focus duration
last_focus_state = None  # Track last state (True for Focused, False for Not Focused)

# Eye Aspect Ratio (EAR) function
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def process_frame(frame):
    global last_focus_state, focus_start_time, total_focus_time

    is_focused = False  # Default to not focused if no face is detected
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    faces = detector(gray)

    if len(faces) == 0:
        is_focused = False
    else:
        for face in faces:
            is_focused = True  # Assume focused initially

            landmarks = predictor(gray, face)
            landmarks = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)])

            # Extract eyes
            left_eye = landmarks[LEFT_EYE]
            right_eye = landmarks[RIGHT_EYE]

            # Compute EAR for both eyes
            left_EAR = eye_aspect_ratio(left_eye)
            right_EAR = eye_aspect_ratio(right_eye)
            avg_EAR = (left_EAR + right_EAR) / 2.0

            # If eyes closed, set as not focused
            if avg_EAR < 0.25:
                is_focused = False

    # **LOG FOCUS CHANGES**
    current_time = time.strftime("%H:%M:%S", time.localtime())
    
    if last_focus_state is None or last_focus_state != is_focused:
        focus_logs.append((current_time, "Focused" if is_focused else "Not Focused"))
        
        if is_focused:
            focus_start_time = time.time()  # Start tracking focus duration
        else:
            if focus_start_time is not None:
                total_focus_time += time.time() - focus_start_time
                focus_start_time = None
    
    last_focus_state = is_focused
    return frame, is_focused

def save_focus_log():
    """Saves the focus log to a CSV file and prints total focus time"""
    df = pd.DataFrame(focus_logs, columns=["Timestamp", "Focus State"])
    df.to_csv("focus_log.csv", index=False)
    print(f"Total Focus Time: {total_focus_time:.2f} seconds")

