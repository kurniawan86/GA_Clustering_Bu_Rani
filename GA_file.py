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
    mut_rate = 0.3
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
            return [m,f]

    def crossover(self):
        male,female = self.selection()
        if self.crOver == "aritmatik" or self.crOver == None:
            crossOver = aritmatik(self.individu[male],self.individu[female])
            child1,child2 = crossOver.getOffspring()
            return child1,child2

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

    def mainProgram(self):

        fitness = []
        for i in range(self.maxloop):
            self.calFitness()
            print("FITNESS ",self.fitness)
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
            fitness.append(min(self.fitness))
            print("min ", min(self.fitness))
            self.individu = new

        self.plot(fitness)
        print("best Fitnestt ", min(self.fitness))
        self.bestFitness = min(self.fitness)
        best_centroid = self.transformToCentroid(self.individu[0])
        print("best Value (individu) \n", best_centroid)
        self.bestCentroid = best_centroid

    def plot(self, val):
        plt.plot(val)
        plt.show()

    def combine(self, offspring):
        unionPop = self.individu+offspring
        fit = self.calFitnessOS(offspring)
        unionFit = self.fitness + fit
        print("fitness asli",self.fitness)
        print("fit offspring ",fit)
        print("unionfit", unionFit)
        s = np.array(unionFit)
        sort_index = np.argsort(s)
        print("sort index :",sort_index)
        new = []
        for i in range(self.npop):
            for j in range(len(sort_index)):
                if i == sort_index[j]:
                    new.append(unionPop[j])
        return new


