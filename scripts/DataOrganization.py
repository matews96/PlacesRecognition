import os
import Auxiliar as aux

def dataOrganization():
    datasetPath = "/Users/Mateo/PycharmProjects/PlacesRecognition/dataset"

    for directoryIndex, directoryName in enumerate(aux.nonOcultedFiles(datasetPath)):
        for imageIndex, imageName in enumerate(aux.getNonCSVFiles(os.path.join(datasetPath, directoryName))):
            extension = aux.getFileExtension(imageName)
            oldPath = os.path.join(datasetPath, directoryName, imageName)
            newPath = os.path.join(datasetPath, directoryName, directoryName + "_" + str(imageIndex) + extension)
            os.rename(oldPath, newPath)
