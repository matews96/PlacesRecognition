import dask.dataframe as dd
import os
from sklearn.cluster import MiniBatchKMeans
import joblib

def descriptorsClustering(clusteringModelsPath):

    print("###################Descriptors Clustering Started##########################")
    print("Loading Data......")
    df = dd.read_csv("/Users/Mateo/PycharmProjects/PlacesRecognition/dataset/*/agora_0.csv", **{'header': None})

    modelCenters = [150, 200, 300]

    for modelCenter in modelCenters:
        print(f"Creating and fitting model {modelCenter}x128...")
        model = MiniBatchKMeans(n_clusters=modelCenter, verbose=True)
        model.fit(df)
        print(f"Saving model {modelCenter}x128...")
        modelName = "kmeans_" + str(modelCenter) + "_128.pkl"
        joblib.dump(model, os.path.join(clusteringModelsPath, modelName))

    print("###################Descriptors Clustering Finished##########################")
