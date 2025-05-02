import cv2
import time
import numpy as np
import mediapipe as mp
from ultralytics import YOLO

# Load YOLOv8 model for phone detection
model = YOLO("yolov8n.pt")

# Global variables
EYE_AR_THRESHOLD = 0.20

# MediaPipe setup for face detection and mesh
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

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

""" Calculate Euclidean distance between two points """
def euclidean_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

""" Calculate Manhattan distance between two points """
def manhattan_distance(point1, point2):
    return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1]) 

""" Calculate the eye aspect ratio from eye landmarks """
def eye_aspect_ratio(eye_points):
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

"""
    Process video frame for face attention and phone detection
    Returns: detection, is_focused flag
"""
def process_image_frame(frame):
    has_face_detected = False
    has_phone_detected = False
    has_head_moved = False
    are_eyes_closed = False 
    gaze = None
    
    is_focused = True  # Start with assumption of focused, will update based on conditions
    
    # Convert frame to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    """///////////////// Face Detection START /////////////////"""

    # Face Detection using MediaPipe
    face_detection_results = face_detection.process(rgb_frame)

    if face_detection_results.detections:
        has_face_detected = True
    else:
        # No face detected - not focused
        is_focused = False

    """///////////////// Face Detection END /////////////////"""

    """ Move to Phone Detection only if a face was detected """
    if has_face_detected:
        """///////////////// Phone Detection START /////////////////"""

        # Phone Detection using YOLO
        has_phone_detected = False
        results = model(rgb_frame)  # Run YOLOv8 model on frame

        for r in results:
            for box in r.boxes:
                class_id = int(box.cls[0])  # Get class index
                label = model.names[class_id]  # Get class name

                if label == "cell phone":  # Detect phone
                    has_phone_detected = True
                    is_focused = False

    """///////////////// Phone Detection END /////////////////"""

    # Face Mesh processing for eye tracking

    """ Move to Pose Evaluation only if a face was detected """
    if has_face_detected:

        mesh_results = face_mesh.process(rgb_frame)
        (h, w) = frame.shape[:2]  # Getting height and width of frame
        """
            IMPORTANT: (h, w) = frame.shape[:2] at the beginning of the function != (h, w) = frame.shape[:2] here
            IDK whether frame got changed in some function or something.
            But now, h and w are correct and hence landmark.x * w and landmark.y * h give correct coordinates over frame.
        """
        if mesh_results.multi_face_landmarks:
            for face_landmarks in mesh_results.multi_face_landmarks:

                """///////////////// Pose Evaluation START /////////////////"""

                model_points = np.array([
                    (0.0, 0.0, 0.0),         # Nose tip
                    (0.0, -63.6, -12.5),     # Chin
                    (-43.3, 32.7, -26.0),    # Left eye left corner
                    (43.3, 32.7, -26.0),     # Right eye right corner
                    (-28.9, -28.9, -24.1),   # Left Mouth corner
                    (28.9, -28.9, -24.1)     # Right mouth corner
                ])

                FACE_POINTS = [1, 152, 263, 33, 287, 57]

                image_points = []
                for idx in FACE_POINTS:
                    landmark = face_landmarks.landmark[idx]
                    x = landmark.x * w
                    y = landmark.y * h
                    image_points.append((x, y))

                """IMPORTANT: image_points must be a numpy array """
                image_points = np.array(image_points)

                # Camera Matrix
                size = frame.shape
                focal_length = size[1]
                center = (size[1] / 2, size[0] / 2)

                camera_matrix = np.array([
                    [focal_length, 0, center[0]],
                    [0, focal_length, center[1]],
                    [0, 0, 1]
                ], dtype="double")

                # Solving PnP to get Rotation Vector
                success, rotation_vector, translation_vector = cv2.solvePnP(
                    model_points, image_points, camera_matrix, distCoeffs=None
                )

                # Converting Rotation Vector to Angles
                rotation_mat, _ = cv2.Rodrigues(rotation_vector)
                pose_mat = cv2.hconcat((rotation_mat, translation_vector))
                _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
                pitch, yaw, roll = euler_angles.flatten()

                pose = None
                up_down = None
                if pitch >= -20 and pitch <= 20:
                    up_down = "Center"
                elif pitch < -20:
                    up_down = "Up"
                else:
                    up_down = "Down"

                left_right = None
                if yaw >= -20 and yaw <= 20:
                    left_right = "Center"
                elif yaw < -20:
                    left_right = "Left"
                else:
                    left_right = "Right"

                if up_down != "Center":
                    pose = up_down
                if left_right != "Center":
                    if pose is not None:
                        pose += "-"+left_right
                    else:
                        pose = left_right

                if pose is not None:
                    has_head_moved = True
                    is_focused = False

                """///////////////// Pose Evaluation END /////////////////"""

                """ Move to EAR only if head pose is Center """

                if not has_head_moved:
                    """///////////////// EAR Evaluation START /////////////////"""

                    # LEFT_EYE landmarks for EAR calculation
                    LEFT_EYE = [362, 385, 387, 263, 373, 380]
                    # RIGHT_EYE landmarks for EAR calculation
                    RIGHT_EYE = [33, 160, 158, 133, 153, 144]
                    
                    # Extract eye landmarks for EAR calculation
                    left_eye_points = []
                    right_eye_points = []
                    
                    for idx in LEFT_EYE:
                        landmark = face_landmarks.landmark[idx]
                        x = int(landmark.x * w)
                        y = int(landmark.y * h)
                        left_eye_points.append((x, y))
                    
                    for idx in RIGHT_EYE:
                        landmark = face_landmarks.landmark[idx]
                        x = int(landmark.x * w)
                        y = int(landmark.y * h)
                        right_eye_points.append((x, y))
                    
                    # Calculate EAR for both eyes
                    left_ear = eye_aspect_ratio(left_eye_points)
                    right_ear = eye_aspect_ratio(right_eye_points)
                    avg_ear = (left_ear + right_ear) / 2.0
                    
                    # Check if eyes are closed based on EAR
                    if avg_ear < EYE_AR_THRESHOLD:
                        are_eyes_closed = True
                        is_focused = False

                    """///////////////// EAR Evaluation END /////////////////"""

                    """ Move to Gaze Evaluation only if eyes are open """

                    if not are_eyes_closed:
                        """///////////////// Gaze Evaluation START /////////////////"""
                        
                        LEFT_IRIS = [474, 475, 476, 477]
                        RIGHT_IRIS = [469, 470, 471, 472]

                        # Extract iris landmarks
                        left_iris = []
                        right_iris = []
                        
                        for idx in LEFT_IRIS:
                            landmark = face_landmarks.landmark[idx]
                            x = int(landmark.x * w)
                            y = int(landmark.y * h)
                            left_iris.append((x, y))
                        
                        for idx in RIGHT_IRIS:
                            landmark = face_landmarks.landmark[idx]
                            x = int(landmark.x * w)
                            y = int(landmark.y * h)
                            right_iris.append((x, y))
                        print("Hello", gaze)
                        # Calculate iris centers
                        def iris_center(iris):
                            x = int(np.mean([pt[0] for pt in iris]))
                            y = int(np.mean([pt[1] for pt in iris]))
                            return x, y

                        l_iris_center = iris_center(left_iris)
                        r_iris_center = iris_center(right_iris)

                        # Gaze logic (Left Eye only)
                        gaze_ratio = (euclidean_distance(l_iris_center, left_eye_points[0]) / (euclidean_distance(left_eye_points[0], left_eye_points[3]) + 1e-6))

                        if gaze_ratio >= 0.50 and gaze_ratio <= 0.70:
                            gaze = "Center"
                        elif gaze_ratio < 0.50:
                            gaze = "Right"
                        else:
                            gaze = "Left"

                        if gaze != "Center":
                            is_focused = False
                        else:
                            gaze = None

                        """///////////////// Gaze Evaluation END /////////////////""" 
    prediction = ""
    if has_phone_detected:
        prediction += "Using Phone"
    elif not has_face_detected:
        prediction += "No Face"
    elif not has_head_moved and are_eyes_closed:
        prediction += "Eyes Closed"
    elif has_head_moved:
        prediction += "Looking Away"
    elif gaze != None:
        prediction += "Gazing Away"
    elif not is_focused:
        prediction += "Not Focused"
    else:
        prediction += "Focused"

    return is_focused, prediction