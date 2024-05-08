import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
index_list = [70, 63, 105, 66, 107, 336, 296, 334, 293, 300, 122, 196, 3, 51, 281, 248, 419, 351, 37, 0, 267, 4, 
                152, 33, 133, 362, 263, 148, 377, 356, 127]

with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.6) as face_mesh:
    image = cv2.imread('images/Banano.jpeg')
    height, width, _ = image.shape
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)
    #print(results.multi_face_landmarks)
    if results.multi_face_landmarks is not None:
        for face_landmarks in results.multi_face_landmarks:
            # mp_drawing.draw_landmarks(image, face_landmarks) #Acceder a todos los puntos solos
            # Punto de la nariz
            '''
            x_NT = int(face_landmarks.landmark[4].x * width)
            y_NT = int(face_landmarks.landmark[4].y * height)
            z_NT = face_landmarks.landmark[4].z
            print('Profundidad de punta de nariz: ', z_NT)
            cv2.circle(image, (x_NT, y_NT), 2, (255, 0, 0), 2)
            '''
            for index in index_list:
                x = int(face_landmarks.landmark[index].x * width)
                y = int(face_landmarks.landmark[index].y * height)
                z = face_landmarks.landmark[index].z
                print('Profundidad de punta de nariz: ', z)
                cv2.circle(image, (x, y), 2, (255, 0, 0), 2)
    cv2.imshow("Imagen", image)
    cv2.waitKey(0)
cv2.destroyAllWindows()
