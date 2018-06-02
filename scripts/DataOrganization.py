import os
import cv2
import Auxiliar as aux
import numpy as np

def dataOrganization(datasetPath, evalPath):
    percentage = 20
    for directoryIndex, directoryName in enumerate(aux.nonOcultedFiles(datasetPath)):

        numberOfImagesInCategory = len(aux.nonOcultedFiles(os.path.join(datasetPath, directoryName)))
        numberOfImagesForEvaluation = int(numberOfImagesInCategory * (percentage/100))
        imagesToEvaluation = np.random.randint(0, numberOfImagesInCategory - 1, numberOfImagesForEvaluation)

        for imageIndex, imageName in enumerate(aux.nonOcultedFiles(os.path.join(datasetPath, directoryName))):
            extension = ".png"
            oldPath = os.path.join(datasetPath, directoryName, imageName)
            newPath = os.path.join(datasetPath, directoryName,
                                   directoryName + "_" + str(imageIndex) + extension)
            if imageIndex in imagesToEvaluation:
                newPath = os.path.join(evalPath, directoryName, directoryName + "_" + str(imageIndex) + extension)

            os.rename(oldPath, newPath)

            img = cv2.imread(newPath)

            if img.shape[0] > 2000:
                img = cv2.resize(img, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_AREA)
            else:
                img = cv2.resize(img, None, fx=1, fy=1, interpolation=cv2.INTER_AREA)

            cv2.imwrite(newPath, img, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
