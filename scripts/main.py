import DataOrganization
import DescriptorsExtraction
import DescriptorsClustering
import Classification

datasetPath = "/Users/Mateo/PycharmProjects/PlacesRecognition/dataset"
clusteringModelsPath = "/Users/Mateo/PycharmProjects/PlacesRecognition/models/clustering"
classificationModelsPath = "/Users/Mateo/PycharmProjects/PlacesRecognition/models/classification"

#This function orders the dataset, it should be a directory that has a directory for each category
#and inside of each should be placed the training images for that category, tha name of the directory
#will be the category label

def createModels():
    DataOrganization.dataOrganization(datasetPath)
    DescriptorsExtraction.descriptorsExtraction(datasetPath)
    DescriptorsClustering.descriptorsClustering(clusteringModelsPath)
    Classification.classification()


def main():
    createModels()


if __name__ == "__main__":
    main()