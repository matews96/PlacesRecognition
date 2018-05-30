import numpy as np
import os
import cv2
import Auxiliar as aux


def descriptorsExtraction():
    datasetPath = "/Users/Mateo/PycharmProjects/PlacesRecognition/dataset"

    sift = cv2.xfeatures2d.SIFT_create()

    numberOfImages = aux.getDatasetSize(datasetPath)
    numberOfCategories = aux.getDatasetNumberOfCategories(datasetPath)
    actualImage = 1

    print("###################Descriptors Extraction Started##########################")
    print("----------------------------------------------------------------------------")

    for directoryIndex, directoryName in enumerate(aux.nonOcultedFiles(datasetPath)):
        for imageIndex, imageName in enumerate(aux.getNonCSVFiles(os.path.join(datasetPath, directoryName))):
            imagePath = os.path.join(datasetPath, directoryName, imageName)
            descriptorsCSVName = aux.changeFileExtension(imageName, ".csv")
            descriptorsCSVPath = os.path.join(datasetPath, directoryName, descriptorsCSVName)

            print(f"Calculating for image {actualImage} of {numberOfImages} in category {directoryName} "
                  f"(Category {directoryIndex + 1} of {numberOfCategories})")

            image = cv2.imread(imagePath, 0)
            print("     finding descriptors...")
            kp, descriptors = sift.detectAndCompute(image, None)
            print("     saving descriptors...")
            np.savetxt(descriptorsCSVPath, descriptors, delimiter=',')

            actualImage += 1

    print("----------------------------------------------------------------------------")

    print("###################Descriptors Extraction Finished##########################")

    print("----------------------------------------------------------------------------")
