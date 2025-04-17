import numpy as np
import cv2
import dlib
from scipy.spatial import distance
import mediapipe as mp

detector = dlib.get_frontal_face_detector()
net = cv2.dnn.readNetFromCaffe('src/deploy.prototxt', 'src/res10_300x300_ssd_iter_140000_fp16.caffemodel')

# predictor = dlib.shape_predictor(r"C:\Users\HP\atten-sense\src\shape_predictor_68_face_landmarks.dat")
predictor = dlib.shape_predictor("src/shape_predictor_68_face_landmarks.dat")

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                  max_num_faces=1,
                                  refine_landmarks=True, # Includes iris landmarks
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(color = (0, 255, 0), thickness=1, circle_radius=1)

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
    (h, w) = frame.shape[:2] # Getting height and width of captured frame

    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # Preprocess input blob for the model
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))  # Mean subtraction
    
    # Pass the blob through the Deep CNN
    net.setInput(blob)
    detections = net.forward()

    results = face_mesh.process(frame)

    # Iterate over all the entities detected
    face_detected = False
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # Confidence threshold
            face_detected = True # Atleast one face detected
            is_focused = True
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            (x1, y1, x2, y2) = box.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{confidence*100:.1f}%"
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            face_dlib_rect = dlib.rectangle(x1, y1, x2, y2)

            # landmarks = predictor(gray, face_dlib_rect)
            # landmarks = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)])
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # mp_drawing.draw_landmarks(
                    #     image=frame,
                    #     landmark_list=face_landmarks,
                    #     connections=mp_face_mesh.FACEMESH_TESSELATION,
                    #     landmark_drawing_spec=drawing_spec,
                    #     connection_drawing_spec=drawing_spec)

                    # Eye landmark indices
                    LEFT_EYE = [362, 385, 387, 263, 373, 380] # list(range(42, 48))
                    RIGHT_EYE = [33, 160, 158, 133, 153, 144] # list(range(36, 42))

                    left_eye = []
                    right_eye = []
                    # Extract eye landmarks
                    for idx in LEFT_EYE:
                        x = int(face_landmarks.landmark[idx].x * w)
                        y = int(face_landmarks.landmark[idx].y * h)
                        left_eye.append((x, y))
                    for idx in RIGHT_EYE:
                        x = int(face_landmarks.landmark[idx].x * w)
                        y = int(face_landmarks.landmark[idx].y * h)
                        right_eye.append((x, y))

                    # Compute EAR for both eyes
                    left_EAR = eye_aspect_ratio(left_eye)
                    right_EAR = eye_aspect_ratio(right_eye)
                    avg_EAR = (left_EAR + right_EAR) / 2.0

                    # Draw eyes
                    # cv2.polylines(frame, [left_eye], True, (0, 255, 0), 1)
                    # cv2.polylines(frame, [right_eye], True, (0, 255, 0), 1)
                    cv2.polylines(frame, [np.array(left_eye, dtype=np.int32).reshape((-1, 1, 2))], True, (0, 255, 0), 1)
                    cv2.polylines(frame, [np.array(right_eye, dtype=np.int32).reshape((-1, 1, 2))], True, (0, 255, 0), 1)

                    # Eye status
                    if avg_EAR < 0.25:
                        cv2.putText(frame, "Eyes Closed", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (135, 206, 235), 2)
                        is_focused = False
                    else:
                        cv2.putText(frame, "Eyes Open", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # # Head Pose Estimation
            # image_points = np.array([
            #     landmarks[30],  # Nose tip
            #     landmarks[36],  # Left eye left corner
            #     landmarks[45],  # Right eye right corner
            #     landmarks[48],  # Left mouth corner
            #     landmarks[54],  # Right mouth corner
            #     landmarks[8]  # Chin
            # ], dtype=np.float32)

            # # 3D model points for head pose estimation
            # MODEL_POINTS = np.array([
            #     (0.0, 0.0, 0.0),  # Nose tip
            #     (-30.0, -125.0, -30.0),  # Left eye left corner
            #     (30.0, -125.0, -30.0),  # Right eye right corner
            #     (-60.0, -70.0, -60.0),  # Left mouth corner
            #     (60.0, -70.0, -60.0),  # Right mouth corner
            #     (0.0, -150.0, -50.0)  # Chin
            # ], dtype=np.float32)

            # # Camera matrix (Assuming a standard focal length)
            # FOCAL_LENGTH = 600
            # CAMERA_MATRIX = np.array([
            #     [FOCAL_LENGTH, 0, 320],
            #     [0, FOCAL_LENGTH, 240],
            #     [0, 0, 1]
            # ], dtype=np.float32)

            # DIST_COEFFS = np.zeros((4, 1))  # Assuming no lens distortion

            # success, rotation_vector, translation_vector = cv2.solvePnP(
            #     MODEL_POINTS, image_points, CAMERA_MATRIX, DIST_COEFFS
            # )

            # if success:
            #     rmat, _ = cv2.Rodrigues(rotation_vector)
            #     angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
            #     yaw = angles[1]  # Left-right head movement

            #     if abs(yaw) > 20:  # If the user is looking away
            #         cv2.putText(frame, "Looking Away", (x1, y2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            #         is_focused = False

    if not face_detected:  # No face detected
        cv2.putText(frame, "No Face Detected", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return frame, is_focused

    return frame, is_focused