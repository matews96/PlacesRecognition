import dask.dataframe as dd
import os
import numpy as np
import os
import cv2
import joblib
from sklearn import svm
import Auxiliar as aux

def classification():
    classificationModelsPath = "/Users/Mateo/PycharmProjects/PlacesRecognition/models/classification"
    clusteringModelsPath = "/Users/Mateo/PycharmProjects/PlacesRecognition/models/clustering"
    datasetPath = "/Users/Mateo/PycharmProjects/PlacesRecognition/dataset"
    numberOfImages = aux.getDatasetSize(datasetPath)
    numberOfCategories = aux.getDatasetNumberOfCategories(datasetPath)
    actualImage = 1

    print("###################Generating classification model##########################")

    for clusteringModelName in aux.nonOcultedFiles(clusteringModelsPath):

        centersNumber = int(clusteringModelName.split("_")[-2])
        X = []
        y = []
        labels = []
        clusteringModel = joblib.load(os.path.join(clusteringModelsPath, clusteringModelName))
        clusteringModel.verbose = False

        for directoryIndex, directoryName in enumerate(aux.nonOcultedFiles(datasetPath)):
            for imageIndex, imageName in enumerate(aux.getCSVFiles(os.path.join(datasetPath, directoryName))):
                print(f"Calculating for image {actualImage} of {numberOfImages} in category {directoryName} "
                      f"(Category {directoryIndex + 1} of {numberOfCategories})")
                image = dd.read_csv(os.path.join(datasetPath, directoryName, imageName), **{'header': None})
                image = clusteringModel.predict(image)
                hist = np.histogram(image, bins=np.array(range(centersNumber)), density=True)
                X.append(hist[0])
                y.append(directoryName)
                actualImage += 1

        print("creating an saving linear model for " + str(centersNumber))
        classificationModel = svm.LinearSVC(verbose=True)
        classificationModel.fit(X, y)
        joblib.dump(classificationModel, classificationModelsPath + "/svm_lineal_" + str(centersNumber) + ".pkl")
        print("creating and saving model for " + str(centersNumber))
        classificationModel = svm.SVC(verbose=True)
        classificationModel.fit(X, y)
        joblib.dump(classificationModel, classificationModelsPath + "/svm_" + str(centersNumber) + ".pkl")

    print("###################clasification model generated##########################")

