import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

class data:
    dset = []
    distMin = []
    index = []
    def __init__(self):
        self.readDataExcel()

    def fitness(self, centroids):
        self.calDatas2centroids(centroids)
        return sum(self.distMin)

    def readDataExcel(self):
        file = pd.read_excel(open('borneo.xlsx', 'rb'))
        dframe = pd.DataFrame(file, columns=(['x', 'y']))
        self.dset = np.array(dframe)

    def plotScatter(self):
        fig, ax = plt.subplots()
        tmp = self.dset
        ax.scatter(tmp[:, 0], tmp[:, 1], marker='o')
        plt.title("Scatter dataset")
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()

    def calDistance(self, centroid, data):
        n = centroid.size
        dis = 0
        for i in range(n):
            dis = dis+math.pow(centroid[i]-data[i], 2)
        return math.sqrt(dis)

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

    def calDatas2centroids(self, centroids):
        n, m = self.dset.shape
        distMin = []
        index = []
        for i in range(n):
            dist, ind = self.calData2Centroids(
                centroids, self.dset[i])
            distMin.append(dist)
            index.append(ind)
        self.distMin = distMin
        self.index = index

    def calSSE(self):
        return sum(self.distMin)

    def getCluster(self, centroids):
        n, m = self.dset.shape
        j, k = centroids.shape
        temp1 = []
        for i in range(k):
            temp2 = []
            for j in range(n):
                if i == self.index[j]:
                    temp2.append(self.dset[j])
            temp1.append(temp2)
        return temp1

    def scatterDataCentroid(self, centroids):
        fig, ax = plt.subplots()
        n,m = centroids.shape
        # print("XXXX ", n , m)
        tes = []
        # for i in range(n):
        #     temp = np.array(centroids[i])
        #     print(temp)
        #     ax.scatter(temp[0], temp[1])
        cent = np.array(centroids)
        tmp = self.dset
        ax.scatter(tmp[:, 0], tmp[:, 1], marker='o')
        plt.legend(tes)
        ax.scatter(cent[:, 0], cent[:, 1], marker='x')
        plt.title('Scatter')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()