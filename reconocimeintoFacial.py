import cv2
import os
from dotenv import load_dotenv


# Es para tomar los nombres de las emprsonas entrenadas
load_dotenv('.env')
dataPath = os.getenv('DataPath')
imagePaths = os.listdir(dataPath)

#face_recognizer = cv2.face.EigenFaceRecognizer_create()
#face_recognizer = cv2.face.FisherFaceRecognizer_create()
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

#Leer modelo
#face_recognizer.read('modeloEigenFace.xml')
#face_recognizer.read('modeloFisherFace.xml')
face_recognizer.read('modeloLBPHFace.xml')

cap = cv2.VideoCapture(1,cv2.CAP_DSHOW)
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    if ret == False : break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = gray.copy()
    faces = faceClassif.detectMultiScale(gray,1.3,5)
    for (x, y, w, h) in faces:
        rostro = auxFrame[y:y+h, x:x+w]
        rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
        result = face_recognizer.predict(rostro)
        cv2.putText(frame, '{}'.format(result), (x, y-5), 1, 1.13, (255, 255, 0), 1, cv2.LINE_AA)
        '''
        # EigenFaces valor más cercano a 0 son más confiables valores en el rango de los miles
        if result[1] < 5700:
            cv2.putText(frame, '{}'.format(imagePaths[result[0]]), (x, y-25), 1, 1.13, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0), 2)
        else:
            cv2.putText(frame, 'Desconocido', (x, y-25), 1, 1.13, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255), 2)
        '''
        '''
        # FisherFaces los vaores más bajos son los más confiables, se empieza en 100
        if result[1] < 500:
            cv2.putText(frame, '{}'.format(imagePaths[result[0]]), (x, y-25), 1, 1.13, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0), 2)
        else:
            cv2.putText(frame, 'Desconocido', (x, y-25), 1, 1.13, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255), 2)
        '''
    cv2.imshow('frame',frame)
    k = cv2.waitKey(1)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()

