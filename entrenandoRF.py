import cv2
import os
import numpy as np
from dotenv import load_dotenv

load_dotenv('.env')
dataPath = os.getenv('DataPath')
peopleList = os.listdir(dataPath)
print('Lista de personas: ', peopleList)

labels = []
facesData = []
label = 0

for nameDir in peopleList:
	personPath = dataPath + '/' + nameDir
	for fileName in os.listdir(personPath):
		labels.append(label)
		facesData.append(cv2.imread(personPath+'/'+fileName,0))
	label = label + 1

# Métodos para entrenar el reconocedor, los métodos estan ordenados del que más tarda al que menos tarda en entrenar
#face_recognizer = cv2.face.EigenFaceRecognizer_create() #Entrena el reconocimiento por einegfaces, todas la imagenes deben de ser del mismo tamaño
#face_recognizer = cv2.face.FisherFaceRecognizer_create()
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# Entrenando el reconocedor de rostros
print("Entrenando...")
face_recognizer.train(facesData, np.array(labels))

# Almacenando el modelo obtenido
#face_recognizer.write('modeloEigenFace.xml')
#face_recognizer.write('modeloFisherFace.xml')
face_recognizer.write('modeloLBPHFace.xml')
print("Modelo almacenado...")

