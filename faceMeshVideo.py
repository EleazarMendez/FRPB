import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
index_list = [70, 63, 105, 66, 107, 336, 296, 334, 293, 300, 122, 196, 3, 51, 281, 248, 419, 351, 37, 0, 267, 4, 
                152, 33, 133, 362, 263, 148, 377, 356, 127]

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.6) as face_mesh:
    while True:
        ret, frame = cap.read()
        if ret == False: break
        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)
        if results.multi_face_landmarks is not None:
            for face_landmarks in results.multi_face_landmarks:
                #mp_drawing.draw_landmarks(frame, face_landmarks)
                for index in index_list:
                    x = int(face_landmarks.landmark[index].x * width)
                    y = int(face_landmarks.landmark[index].y * height)
                    z = face_landmarks.landmark[index].z
                    cv2.circle(frame, (x, y), 2, (255, 0, 0), 2)
        cv2.imshow('Frame', frame)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
cap.release()
cv2.destroyAllWindows()

