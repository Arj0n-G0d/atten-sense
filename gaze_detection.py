import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
mp_drawing = mp.solutions.drawing_utils

# Eye and iris landmark indices
LEFT_EYE = [33, 133]
RIGHT_EYE = [362, 263]
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = frame.shape

            # Get landmarks
            def get_coords(indices):
                return [face_landmarks.landmark[i] for i in indices]

            left_eye = get_coords(LEFT_EYE)
            right_eye = get_coords(RIGHT_EYE)
            left_iris = get_coords(LEFT_IRIS)
            right_iris = get_coords(RIGHT_IRIS)

            # Calculate iris centers
            def iris_center(iris):
                x = int(np.mean([pt.x for pt in iris]) * w)
                y = int(np.mean([pt.y for pt in iris]) * h)
                return x, y

            l_center = iris_center(left_iris)
            r_center = iris_center(right_iris)

            # Draw circles
            cv2.circle(frame, l_center, 2, (0, 255, 0), -1)
            cv2.circle(frame, r_center, 2, (0, 255, 0), -1)

            # Gaze logic (Left Eye only)
            l_x1 = int(left_eye[0].x * w)
            l_x2 = int(left_eye[1].x * w)
            ratio = (l_center[0] - l_x1) / (l_x2 - l_x1 + 1e-6)

            if ratio < 0.35:
                gaze = "Right"
            elif ratio > 0.65:
                gaze = "Left"
            else:
                gaze = "Center"

            cv2.putText(frame, f"Gaze: {gaze}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow('Gaze Detection', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
