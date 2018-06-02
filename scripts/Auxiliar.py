import os

def getDatasetSize(path):
    numberOfImages = 0
    for directoryName in nonOcultedFiles(path):
        numberOfImages += len(getNonCSVFiles(os.path.join(path, directoryName)))
    return numberOfImages

def getDatasetNumberOfCategories(path):
    return len(nonOcultedFiles(path))

def nonOcultedFiles(path, numericSort = False):
    if numericSort:
        return sorted([file for file in os.listdir(path) if not file.startswith('.')], key=getImageNumber)
    else:
        return sorted([file for file in os.listdir(path) if not file.startswith('.')])

def getCSVFiles(path):
    return [file for file in nonOcultedFiles(path, True) if getFileExtension(file) == ".csv"]

def getNonCSVFiles(path):
    return [file for file in nonOcultedFiles(path, True) if not (getFileExtension(file) == ".csv")]

def getFileExtension(fileName):
    return os.path.splitext(fileName)[-1].lower()

def removeFileExtension(fileName):
    return os.path.splitext(fileName)[0].lower()

def changeFileExtension(fileName, newExtension):
    oldExtension = getFileExtension(fileName)
    return fileName.replace(oldExtension, newExtension)

def getImageNumber(fileName):
    return int(removeFileExtension(fileName).split("_")[1])

def getImageName(fileName):
    return removeFileExtension(fileName).split("_")[0]

def getFileName(path):
    return path.split("/")[-1]
