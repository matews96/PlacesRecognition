import DataOrganization
import DescriptorsExtraction
import DescriptorsClustering
import Classification

def main():

    DataOrganization.dataOrganization()
    DescriptorsExtraction.descriptorsExtraction()
    DescriptorsClustering.descriptorsClustering()
    Classification.classification()

if __name__ == "__main__":
    main()