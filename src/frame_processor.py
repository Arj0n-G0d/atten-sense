import numpy as np
import cv2
import dlib

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("/home/g0d/Desktop/atten-sense/src/shape_predictor_68_face_landmarks.dat")

def compute_euclidean_dist(ptA, ptB):
    ptA, ptB = np.array(ptA), np.array(ptB)  # Convert to NumPy arrays
    dist = np.linalg.norm(ptA - ptB)
    return dist

def process_frame(frame) :
    is_focused = True
    gray = frame.copy()
    gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)

    faces = detector(gray)
    if len(faces) == 0 :
        is_focused = False
    for face in faces :
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()

        # cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        landmarks = predictor(gray, face)

        for n in range(0, 68):
            x, y = landmarks.part(n).x, landmarks.part(n).y 
            # cv2.circle(frame, (x, y), 1, (255, 255, 255), -1)

        a = (landmarks.part(30).x, landmarks.part(30).y)
        b = (landmarks.part(14).x, landmarks.part(14).y)
        c = (landmarks.part(2).x, landmarks.part(2).y)

        if compute_euclidean_dist(a, b) < 20 or compute_euclidean_dist(a, c) < 20 :
            is_focused = False

    return frame, is_focused