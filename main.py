from objective_file import Clustering
import numpy as np
from GA_file import *

if __name__ == '__main__':
    print("Hello GA Clustering")
    f_fitness = Clustering()
    npop = 5
    ndim = 2
    nCluster = 3
    maxloop = 20
    centroids = np.array([[5, 5], [3, 1], [6, 6]])
    obj = standart_GA(npop, ndim, nCluster, maxloop, fitness_function=f_fitness)
    obj.mainProgram()
    f_fitness.plottingScatter(obj.bestCentroid)
