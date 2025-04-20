import cv2
import time
import numpy as np
import mediapipe as mp
from ultralytics import YOLO

# Load YOLOv8 model for phone detection
model = YOLO("yolov8n.pt")

# Constants
PHONE_USAGE_THRESHOLD = 0.5 # Seconds before marking as "Using Phone"
BLINK_THRESHOLD = 0.25  # Seconds - maximum allowed blink duration before considered unfocused
EYE_AR_THRESHOLD = 0.35  # Eye aspect ratio threshold for determining closed eyes

# Global variables for tracking
phone_detected_time = 0
blink_start_time = 0
is_blinking = False
eyes_closed_status = False  # New global variable to track eye closure status

# MediaPipe setup for face detection and mesh
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Initialize face detection and face mesh
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,  # Includes iris landmarks
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Drawing specifications
drawing_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)

def euclidean_distance(point1, point2):
    """Calculate Euclidean distance between two points"""
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def eye_aspect_ratio(eye_points):
    """Calculate the eye aspect ratio from eye landmarks"""
    # Compute the euclidean distances between the horizontal landmarks
    A = euclidean_distance(eye_points[1], eye_points[5])
    B = euclidean_distance(eye_points[2], eye_points[4])
    
    # Compute the euclidean distance between the vertical landmarks
    C = euclidean_distance(eye_points[0], eye_points[3])
    
    # Avoid division by zero
    if C == 0:
        return 0
    
    # Compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    return ear

def process_frame(frame):
    """
    Process frame for face attention and phone detection.
    Returns: processed frame, is_focused flag
    """
    global phone_detected_time, blink_start_time, is_blinking, eyes_closed_status
    
    is_focused = True  # Start with assumption of focused, will update based on conditions
    (h, w) = frame.shape[:2]  # Getting height and width of captured frame
    
    # Convert frame to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Phone Detection using YOLO
    phone_detected = False
    results = model(rgb_frame)  # Run YOLOv8 model on frame

    for r in results:
        for box in r.boxes:
            class_id = int(box.cls[0])  # Get class index
            label = model.names[class_id]  # Get class name

            if label == "cell phone":  # Detect phone
                phone_detected = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, "Phone Detected", (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Check phone usage time
    if phone_detected:
        if phone_detected_time == 0:
            phone_detected_time = time.time()
        elif time.time() - phone_detected_time > PHONE_USAGE_THRESHOLD:
            cv2.putText(frame, "Using Phone!", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            # Mark as not focused when using phone
            is_focused = False
    else:
        phone_detected_time = 0  # Reset timer if phone is not detected

    # Face Detection using MediaPipe
    face_detection_results = face_detection.process(rgb_frame)
    face_detected = False

    if face_detection_results.detections:
        face_detected = True
        for detection in face_detection_results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                         int(bboxC.width * iw), int(bboxC.height * ih)
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Display confidence score
            score = detection.score[0]
            cv2.putText(frame, f"{score:.2f}", (x, y - 10),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    else:
        # No face detected - not focused
        cv2.putText(frame, "No Face Detected", (10, 40), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        is_focused = False
        # Reset eye tracking when no face is detected
        eyes_closed_status = False
        is_blinking = False
        blink_start_time = 0

    # Eyes tracking state (default to previous known state)
    eyes_are_closed = eyes_closed_status
    
    # Face Mesh processing for eye tracking
    if face_detected:
        mesh_results = face_mesh.process(rgb_frame)
        
        if mesh_results.multi_face_landmarks:
            for face_landmarks in mesh_results.multi_face_landmarks:
                # Eye landmark indices for MediaPipe Face Mesh
                # LEFT_EYE landmarks for EAR calculation
                LEFT_EYE = [362, 385, 387, 263, 373, 380]
                # RIGHT_EYE landmarks for EAR calculation
                RIGHT_EYE = [33, 160, 158, 133, 153, 144]
                
                # Additional landmarks for drawing eye contours (LEFT EYE)
                LEFT_EYE_CONTOUR = [
                    # Upper eyelid
                    362, 382, 381, 380, 374, 373, 390, 249, 263,
                    # Lower eyelid
                    466, 388, 387, 386, 385, 384, 398, 362
                ]
                
                # Additional landmarks for drawing eye contours (RIGHT EYE)
                RIGHT_EYE_CONTOUR = [
                    # Upper eyelid
                    33, 7, 163, 144, 145, 153, 154, 155, 133,
                    # Lower eyelid
                    246, 161, 160, 159, 158, 157, 173, 33
                ]
                
                # Extract eye landmarks for EAR calculation
                left_eye_points = []
                right_eye_points = []
                
                for idx in LEFT_EYE:
                    landmark = face_landmarks.landmark[idx]
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    left_eye_points.append((x, y))
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
                
                for idx in RIGHT_EYE:
                    landmark = face_landmarks.landmark[idx]
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    right_eye_points.append((x, y))
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
                
                # Extract eye contour landmarks for drawing
                left_contour_points = []
                right_contour_points = []
                
                for idx in LEFT_EYE_CONTOUR:
                    landmark = face_landmarks.landmark[idx]
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    left_contour_points.append((x, y))
                
                for idx in RIGHT_EYE_CONTOUR:
                    landmark = face_landmarks.landmark[idx]
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    right_contour_points.append((x, y))
                
                # Draw eye contours with detailed lines
                cv2.polylines(frame, [np.array(left_contour_points)], True, (255, 105, 65), 2)
                cv2.polylines(frame, [np.array(right_contour_points)], True, (255, 105, 65), 2)
                
                # Calculate EAR for both eyes
                left_ear = eye_aspect_ratio(left_eye_points)
                right_ear = eye_aspect_ratio(right_eye_points)
                avg_ear = (left_ear + right_ear) / 2.0
                
                # Display EAR value for debugging
                cv2.putText(frame, f"EAR: {avg_ear:.2f}", (10, 110), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Check if eyes are closed based on EAR
                if avg_ear < EYE_AR_THRESHOLD:
                    eyes_are_closed = True
                    # Update global tracking variable
                    eyes_closed_status = True
                    
                    # Start or continue tracking blink duration
                    if not is_blinking:
                        blink_start_time = time.time()
                        is_blinking = True
                    
                    # Calculate how long eyes have been closed
                    blink_duration = time.time() - blink_start_time
                    
                    # Check if eyes closed longer than threshold
                    if blink_duration > BLINK_THRESHOLD:
                        cv2.putText(frame, f"Eyes Closed ({blink_duration:.1f}s)", (10, 80), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        # Mark as not focused for prolonged eye closure
                        is_focused = False
                    else:
                        cv2.putText(frame, f"Blinking ({blink_duration:.1f}s)", (10, 80), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
                else:
                    eyes_are_closed = False
                    eyes_closed_status = False
                    is_blinking = False
                    blink_start_time = 0
                    cv2.putText(frame, "Eyes Open", (10, 80), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Handle case when face is detected but no eye landmarks found
    if face_detected and eyes_are_closed and is_blinking:
        blink_duration = time.time() - blink_start_time
        if blink_duration > BLINK_THRESHOLD:
            is_focused = False

    # Add overall status display to frame
    status_text = "Status: "
    if phone_detected and time.time() - phone_detected_time > PHONE_USAGE_THRESHOLD:
        status_text += "Using Phone (Not Focused)"
        status_color = (0, 0, 255)  # Red
    elif not face_detected:
        status_text += "No Face (Not Focused)"
        status_color = (0, 0, 255)  # Red
    elif eyes_are_closed and is_blinking and (time.time() - blink_start_time > BLINK_THRESHOLD):
        status_text += "Eyes Closed (Not Focused)"
        status_color = (0, 0, 255)  # Red
    elif not is_focused:
        status_text += "Not Focused"
        status_color = (0, 0, 255)  # Red
    else:
        status_text += "Focused"
        status_color = (0, 255, 0)  # Green
    
    # Debug information - show blink duration
    if is_blinking:
        blink_duration = time.time() - blink_start_time
        cv2.putText(frame, f"Blink time: {blink_duration:.2f}s", (10, 140), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
    cv2.putText(frame, status_text, (10, frame.shape[0] - 20), 
              cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

    return frame, is_focused

