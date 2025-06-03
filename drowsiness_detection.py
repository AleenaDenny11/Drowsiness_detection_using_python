import cv2
import mediapipe as mp
import numpy as np
import time
import winsound 


mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                   max_num_faces=1,
                                   min_detection_confidence=0.5,
                                   min_tracking_confidence=0.5)


LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [263, 387, 385, 362, 380, 373]


def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def compute_ear(eye):
    A = euclidean(eye[1], eye[5])
    B = euclidean(eye[2], eye[4])
    C = euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


cap = cv2.VideoCapture(0)

counter = 0
ALERT_THRESHOLD = 15  
EAR_THRESHOLD = 0.21

print("Press 'Q' to quit.")
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            
            mesh_points = [(int(p.x * w), int(p.y * h)) for p in face_landmarks.landmark]

            
            left_eye = [mesh_points[i] for i in LEFT_EYE]
            right_eye = [mesh_points[i] for i in RIGHT_EYE]

            left_ear = compute_ear(left_eye)
            right_ear = compute_ear(right_eye)
            avg_ear = (left_ear + right_ear) / 2.0

            
            for pt in left_eye + right_eye:
                cv2.circle(frame, pt, 2, (0, 255, 0), -1)

            
            if avg_ear < EAR_THRESHOLD:
                counter += 1
            else:
                counter = 0

            if counter > ALERT_THRESHOLD:
                cv2.putText(frame, "DROWSINESS DETECTED!", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                winsound.Beep(1000, 500)  # Alert sound

    cv2.imshow("Driver Drowsiness Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
