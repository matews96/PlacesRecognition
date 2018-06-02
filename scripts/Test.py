import cv2
import joblib
import os
import numpy as np
import Auxiliar as aux

sift = cv2.xfeatures2d.SIFT_create()


clusmodel = joblib.load("/Users/mateo.echeverri/PycharmProjects/PlacesRecognition/models/clustering/kmeans_150_128.pkl")
clusmodel.verbose = False
classmodel = joblib.load("/Users/mateo.echeverri/PycharmProjects/PlacesRecognition/models/classification/svm_lineal_150.pkl")
evalPath = "/Users/mateo.echeverri/PycharmProjects/PlacesRecognition/eval"


for directoryIndex, directoryName in enumerate(aux.nonOcultedFiles(evalPath)):
    accuracy = 0
    for imageIndex, imageName in enumerate(aux.nonOcultedFiles(os.path.join(evalPath, directoryName))):
        img = cv2.imread(os.path.join(evalPath, directoryName, imageName), 0)
        kp, descriptors = sift.detectAndCompute(img, None)
        img = clusmodel.predict(descriptors)
        hist = np.histogram(img, bins=np.array(range(150)), density=True)
        if classmodel.predict([hist[0]])[0] == directoryName:
            accuracy = accuracy + 1

    accuracy = (accuracy / len(aux.nonOcultedFiles(os.path.join(evalPath, directoryName)))) * 100
    print(f"Acuraccy for {directoryName}: {accuracy}%")


