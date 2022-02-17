import numpy as np
from selection_file import rw
from crossover_file import aritmatik
from mutation_file import *
import matplotlib.pyplot as plt
import math

class GA:
    crossover = None
    f_fit = None
    select = None
    mut = None
    __bound = None
    ndim = 0
    npop = 0
    individu = []
    fitness = []
    fitnesses = []
    mut_rate = 0.2
    cr_rate = 0.6
    maxloop = 0
    nCluster = 0

    def __init__(self, n_popoulation, n_dimension, n_cluster, maxloop, fitness_function=None, selection=None, crossover=None, mutation=None, bound=None):
        self.maxloop = maxloop
        self.nCluster = n_cluster
        self.f_fit = fitness_function
        self.npop = n_popoulation
        self.ndim = n_dimension
        self.__bound = bound
        self.select = selection
        self.mut = mutation
        self.crOver = crossover
        self.__initPosition()

    def __initPositionNoBound(self):
        gen = []
        dim = self.ndim * self.nCluster
        for i in range(dim):
            gen.append(random.random())
        return gen

    def __initPositionBound(self):
        gen = []
        dim = self.ndim * self.nCluster
        mini = self.__bound[0]
        maxi = self.__bound[1]
        for i in range(dim):
            ind = random.randint(mini, maxi-1)
            gen.append(ind)
        return gen

    def __initPosition(self):
        if self.__bound == None:
            for i in range(self.npop):
                self.individu.append(self.__initPositionNoBound())
        else:
            for i in range(self.npop):
                self.individu.append(self.__initPositionBound())

    def transformToCentroid(self, individu):
        arr_1d = np.array(individu)
        arr_2d = np.reshape(arr_1d, (self.nCluster, self.ndim))
        return arr_2d

    def calFitness(self):
        fitnesse = []
        for i in range(self.npop):
            centroid = np.array(self.individu[i])
            centroid = self.transformToCentroid(centroid)
            fitnesse.append(self.f_fit.fitness(centroid))
        self.fitness = fitnesse

    def calFitnessOS(self, offspring):
        fit = []
        for i in range(len(offspring)):
            centroid = np.array(offspring[i])
            centroid = self.transformToCentroid(centroid)
            fit.append(self.f_fit.fitness(centroid))
        return fit

    def selection(self):
        if self.select == "rw" or self.select == None:
            selection = rw(self.individu, self.fitness)
            m,f = selection.pick2Parents()
            return [m, f]

    def crossover(self):
        male,female = self.selection()
        if self.crOver == "aritmatik" or self.crOver == None:
            crossOver = aritmatik(self.individu[male],self.individu[female])
            child1, child2 = crossOver.getOffspring()
            return child1, child2

    def mutation(self):
        if self.mut == "random" or self.mut == None:
            dim = self.ndim * self.nCluster
            mut = RandomUniform(dim)
            return mut.getOffsping()

class standart_GA(GA):
    bestCentroid = None
    bestFitness = None

    def __init__(self, n_popoulation,n_dimension, ncluster, maxloop,fitness_function=None, selection=None, crossover=None, mutation=None, bound=None):
        super(standart_GA, self).__init__(n_popoulation,n_dimension, ncluster, maxloop,fitness_function, selection, crossover, mutation, bound)
        print("hello GA Standar Cluster")

    def mainProgram(self):
        fitness = []
        for loop in range(self.maxloop):
            self.calFitness()
            fitness.append(min(self.fitness))
            offspring = []
            #crossover
            ind = math.ceil((self.cr_rate * self.npop) / 2)
            for i in range(ind):
                child1, child2 = GA.crossover(self)
                offspring.append(child1)
                offspring.append(child2)

            #mutation process
            ind = math.ceil((self.mut_rate * self.npop) / 2)
            for i in range(ind):
                offspring.append(self.mutation())

            #combine
            new = self.combine(offspring)
            self.individu = new

            #pengujian
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

        self.plot(fitness)
        # print("best Fitnestt ", min(self.fitness))
        self.bestFitness = min(self.fitness)
        best_centroid = self.transformToCentroid(self.individu[0])
        # print("best Value (individu) \n", best_centroid)
        self.bestCentroid = best_centroid

    def plot(self, val):
        plt.plot(val)
        plt.show()

    def combine(self, offspring):
        fit = self.calFitnessOS(offspring)
        unionPop = self.individu+offspring
        unionFit = self.fitness + fit
        # un = np.zeros(len(unionFit))
        un = np.array(unionFit)
        # print("unionfit ", unionFit)
        arr, sort_index = self.bubbleSort(un)
        # print("SORT index :",sort_index)
        # self.fitness = arr[0:self.npop]
        # print("arr ", arr)
        # print("union pop :\n", np.array(unionPop))
        # sort_index = np.argsort(s)

        new = []
        for i in range(len(self.fitness)):
            # print("iiiise", i)
            ind = 0
            for j in range(len(unionFit)):
                # print("Jjejej =",j)
                # print("arr ", arr[i])
                # print("unioinFit", unionFit[j])
                if arr[i] == unionFit[j]:
                    # print(type(self.fitness[i]),arr[i])
                    # print(type(unionFit[j]),unionFit[j])
                    # print(arr[i] == unionFit[j])
                    ind = j
                    # print("ind ", ind)
                    break
            new.append(unionPop[ind])
        # print("neww ", new)
        return new

    def bubbleSort(self, arr):
        n = len(arr)
        ind = []
        for i in range(n):
            ind.append(i)

        for i in range(n):
            min = arr[i]
            min1 = ind[i]
            index = i
            for j in range(i+1, n):
                if min > arr[j]:
                    min = arr[j]
                    min1 = ind[j]
                    index = j
            temp = ind[index]
            arr[i], arr[index] = min, arr[i]
            ind[index] = ind[i]
            ind[i] = temp
        return arr, ind

    def sortinge(self, matrix):
        li = []
        for i in range(len(matrix)):
            li.append([matrix[i], i])
        li.sort()
        ss = []

        for x in li:
            ss.append(x[1])
        print(ss)

        for i in range(len(matrix)):
            for j in range (len(matrix)):
                if i == ss[j]:
                    print(matrix[j])

class GA_mean(GA):
    dataset = None
    bestCentroid = None
    bestFitness = None

    def __init__(self, data, n_popoulation,n_dimension, ncluster, maxloop,fitness_function=None, selection=None, crossover=None, mutation=None, bound=None):
        super(GA_mean, self).__init__(n_popoulation,n_dimension, ncluster, maxloop,fitness_function, selection, crossover, mutation, bound)
        print("hello GA Mean Cluster")
        self.dataset = data

    def mainProgram(self):
        fitness = []
        for loop in range(self.maxloop):
            self.calFitness()
            fitness.append(min(self.fitness))
            offspring = []
            #crossover
            ind = math.ceil((self.cr_rate * self.npop) / 2)
            for i in range(ind):
                child1, child2 = GA.crossover(self)
                offspring.append(child1)
                offspring.append(child2)

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