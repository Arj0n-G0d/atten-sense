import numpy as np
import cv2
import dlib
from scipy.spatial import distance

detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor(r"C:\Users\HP\atten-sense\src\shape_predictor_68_face_landmarks.dat")
predictor = dlib.shape_predictor("src/shape_predictor_68_face_landmarks.dat")

# Eye landmark indices
LEFT_EYE = list(range(42, 48))
RIGHT_EYE = list(range(36, 42))

# 3D model points for head pose estimation
MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),  # Nose tip
    (-30.0, -125.0, -30.0),  # Left eye left corner
    (30.0, -125.0, -30.0),  # Right eye right corner
    (-60.0, -70.0, -60.0),  # Left mouth corner
    (60.0, -70.0, -60.0),  # Right mouth corner
    (0.0, -150.0, -50.0)  # Chin
], dtype=np.float32)

# Camera matrix (Assuming a standard focal length)
FOCAL_LENGTH = 600
CAMERA_MATRIX = np.array([
    [FOCAL_LENGTH, 0, 320],
    [0, FOCAL_LENGTH, 240],
    [0, 0, 1]
], dtype=np.float32)

DIST_COEFFS = np.zeros((4, 1))  # Assuming no lens distortion

# Eye Aspect Ratio (EAR) function
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def compute_euclidean_dist(ptA, ptB):
    return np.linalg.norm(np.array(ptA) - np.array(ptB))

def process_frame(frame):
    is_focused = False  # Default to not focused if no face is detected
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    faces = detector(gray)

    if len(faces) == 0:  # No face detected
        cv2.putText(frame, "No Face Detected", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return frame, is_focused

    for face in faces:
        is_focused = True  # Face detected, assume focused until proven otherwise

        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Face box

        landmarks = predictor(gray, face)
        landmarks = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)])

        # Extract eye landmarks
        left_eye = landmarks[LEFT_EYE]
        right_eye = landmarks[RIGHT_EYE]

        # Compute EAR for both eyes
        left_EAR = eye_aspect_ratio(left_eye)
        right_EAR = eye_aspect_ratio(right_eye)
        avg_EAR = (left_EAR + right_EAR) / 2.0

        # Draw eyes
        cv2.polylines(frame, [left_eye], True, (0, 255, 0), 1)
        cv2.polylines(frame, [right_eye], True, (0, 255, 0), 1)

        # Eye status
        if avg_EAR < 0.25:
            cv2.putText(frame, "Eyes Closed", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (135, 206, 235), 2)
            is_focused = False
        else:
            cv2.putText(frame, "Eyes Open", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Check face movement
        nose = (landmarks[30][0], landmarks[30][1])
        right_edge = (landmarks[14][0], landmarks[14][1])
        left_edge = (landmarks[2][0], landmarks[2][1])

        if compute_euclidean_dist(nose, right_edge) < 20 or compute_euclidean_dist(nose, left_edge) < 20:
            is_focused = False

        # Head Pose Estimation
        image_points = np.array([
            landmarks[30],  # Nose tip
            landmarks[36],  # Left eye left corner
            landmarks[45],  # Right eye right corner
            landmarks[48],  # Left mouth corner
            landmarks[54],  # Right mouth corner
            landmarks[8]  # Chin
        ], dtype=np.float32)

        success, rotation_vector, translation_vector = cv2.solvePnP(
            MODEL_POINTS, image_points, CAMERA_MATRIX, DIST_COEFFS
        )

        if success:
            rmat, _ = cv2.Rodrigues(rotation_vector)
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
            yaw = angles[1]  # Left-right head movement

            if abs(yaw) > 20:  # If the user is looking away
                cv2.putText(frame, "Looking Away", (x1, y2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                is_focused = False

    return frame, is_focused