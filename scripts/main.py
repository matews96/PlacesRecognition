import DataOrganization
import DescriptorsExtraction
import DescriptorsClustering
import Classification

datasetPath = "/Users/mateo.echeverri/PycharmProjects/PlacesRecognition/dataset"
evalPath = "/Users/mateo.echeverri/PycharmProjects/PlacesRecognition/eval"
clusteringModelsPath = "/Users/mateo.echeverri/PycharmProjects/PlacesRecognition/models/clustering"
classificationModelsPath = "/Users/mateo.echeverri/PycharmProjects/PlacesRecognition/models/classification"


def createModels():
    #DataOrganization.dataOrganization(datasetPath, evalPath)
    DescriptorsExtraction.descriptorsExtraction(datasetPath)
    DescriptorsClustering.descriptorsClustering(clusteringModelsPath)
    Classification.classification(datasetPath, clusteringModelsPath, classificationModelsPath)


def main():
    createModels()


if __name__ == "__main__":
    main()