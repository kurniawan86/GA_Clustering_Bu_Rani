from GA_file import GA
import numpy as np
from selection_file import rw
from crossover_file import aritmatik
from mutation_file import *
import matplotlib.pyplot as plt
import math

class GA_poly(GA):
    dataset = None
    bestCentroid = None
    bestFitness = None

    def __init__(self, data, n_popoulation,n_dimension, ncluster, maxloop,fitness_function=None, selection=None, crossover=None, mutation=None, bound=None):
        super(GA_poly, self).__init__(n_popoulation,n_dimension, ncluster, maxloop,fitness_function, selection, crossover, mutation, bound)
        print("hello GA POLYGAMi Mean Cluster")
        self.dataset = data

    def mainProgram(self):
        fitness = []
        for loop in range(self.maxloop):
            self.calFitness()
            fitness.append(min(self.fitness))
            offspring = []

            #crossover
            nMate = 4
            ind = math.ceil((self.cr_rate * self.npop) / 2)
            loopMate = math.ceil(ind / nMate)
            # print("loop mate :",loopMate)
            for i in range(loopMate):
                for j in range(nMate):
                    male, female = self.selection()
                    while male==female:
                        # print("male female :", male, female)
                        female = random.randint(0,self.npop-1)
                        # male, female = self.selection()
                    crossOver = aritmatik(self.individu[male], self.individu[female])
                    child1, child2 = crossOver.getOffspring()
                    # print("child 1 ",child1)
                    # print("child 2 ",child2)
                    offspring.append(child1)
                    offspring.append(child2)
            #
            # for i in range(ind):
            #     child1, child2 = GA.crossover(self)
            #     offspring.append(child1)
            #     offspring.append(child2)

            #mutation process
            ind = math.ceil((self.mut_rate * self.npop) / 2)
            for i in range(ind):
                offspring.append(self.mutation())

            #combine
            new = self.combine(offspring)
            # print("tes centroid ", self.transformToCentroid(self.individu[0]))
            centroid_temp = self.transformToCentroid(self.individu[0])
            add = self.get_MeanCentroid(centroid_temp)
            # print("addddd", add[0])
            # print("add ", add)
            # print("before add ", self.individu)

            # print("before add ", self.individu)
            if type(add) == int:
                self.individu = new
            else:
                self.individu = new
                add = add.tolist()
                self.individu[self.npop-1] = add[0]
            # print("after add ", self.individu)

            # pengujian
            # print(loop)
            if loop > 10:
                # print(loop)
                mn = sum(fitness[loop - 10:loop])
                mnn = mn / len(fitness[loop - 10:loop])
                mnm = mnn / fitness[loop]
                xx = fitness[loop]
                xy = fitness[loop-9]
                # print(mnm)
                if xx == xy:
                    print("iterasi stop :", loop)
                    break

        # self.plot(fitness)
        self.bestFitness = min(self.fitness)
        best_centroid = self.transformToCentroid(self.individu[0])
        self.bestCentroid = best_centroid

    def get_MeanCentroid(self, centroid_temp):
        index = self.calDatas2centroids(centroid_temp)
        m, n = centroid_temp.shape
        k, l = self.dataset.shape
        clusters = []
        for i in range(m):
            cluster = []
            for j in range(k):
                if i == index[j]:
                    cluster.append(self.dataset[j].tolist())
            cl = self.get_mean_acluster(np.array(cluster))
            clusters.append(cl)
        cl = np.array(clusters)
        # print("RSSS ", clusters.size)
        dim = self.ndim * self.nCluster
        if cl.size == dim:
            #ini bikin error
            result = cl.reshape([1, dim])
            return result
        else:
            return 0

    def get_mean_acluster(self, data):
        clus = []
        if data.size != 0:
            a,b = data.shape
            for i in range(b):
                clus.append(np.mean(data[:,i]))
        return clus

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

    def calDatas2centroids(self, centroids):
        # centroids = self.transformToCentroid(self.centroids)
        # print("centroid def cal", centroids)
        n, m = self.dataset.shape
        distMin = []
        index = []
        for i in range(n):
            dist, ind = self.calData2Centroids(
                centroids, self.dataset[i])
            distMin.append(dist)
            index.append(ind)
        self.distMin = distMin
        return index

    def plot(self, val):
        plt.plot(val)
        plt.show()

    def combine(self, offspring):
        fit = self.calFitnessOS(offspring)
        unionPop = self.individu+offspring
        unionFit = self.fitness + fit
        un = np.array(unionFit)
        arr, sort_index = self.bubbleSort(un)

        new = []
        for i in range(len(self.fitness)):
            # print("iiiise", i)
            ind = 0
            for j in range(len(unionFit)):
                if arr[i] == unionFit[j]:
                    ind = j
                    break
            new.append(unionPop[ind])
        return new

    def bubbleSort(self, arr):
        n = len(arr)
        ind = []
        for i in range(n):
            ind.append(i)

        for i in range(n):
            min = arr[i]
            index = i
            for j in range(i+1, n):
                if min > arr[j]:
                    min = arr[j]
                    index = j
            temp = ind[index]
            arr[i], arr[index] = min, arr[i]
            ind[index] = ind[i]
            ind[i] = temp
        return arr, ind