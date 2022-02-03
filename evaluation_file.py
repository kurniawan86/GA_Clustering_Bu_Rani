import numpy as np
import math
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score

class evaluation:
    centroids = []
    data = []
    distMin = []
    index = []

    def __init__(self, data, bestCluster):
        self.data = data
        self.centroids = self.transformToCentroid(bestCluster)
        self.calDatas2centroids()

    def get_davies_bouldin(self):
        z = davies_bouldin_score(self.data, self.index)
        print("Davies Boundies  :",z)

    def get_SSE(self):
        print("SSE              :",sum(self.distMin))

    def get_silhouette_score(self):
        z = silhouette_score(self.data, self.index)
        print("Silhoutte Score  :",z)

    def transformToCentroid(self, individu):
        n, m = individu.shape
        arr_1d = np.array(individu)
        arr_2d = np.reshape(arr_1d, (n, m))
        return arr_2d

    def calDistance(self, centroid, data):
        n = centroid.size
        dis = 0
        for i in range(n):
            dis = dis+math.pow(centroid[i]-data[i], 2)
        return dis

    def calData2Centroids(self, centroids, data):
        n, m = centroids.shape
        minimal = self.calDistance(centroids[0], data)
        indMin = 0
        for i in range(n):
            minimal1 = self.calDistance(centroids[i], data)
            if minimal1 < minimal:
                minimal = minimal1
                indMin = i
        return minimal, indMin

    def calDatas2centroids(self):
        centroids = self.centroids
        n, m = self.data.shape
        distMin = []
        index = []
        for i in range(n):
            dist, ind = self.calData2Centroids(
                centroids, self.data[i])
            distMin.append(dist)
            index.append(ind)
        self.distMin = distMin
        self.index = index