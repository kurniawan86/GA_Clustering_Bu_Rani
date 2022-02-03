from objective_file import Clustering
import numpy as np
import pandas as pd
from evaluation_file import evaluation
from GA_file import *

def readDataExcel():
    file = pd.read_excel(open('dataset.xlsx', 'rb'))
    dframe = pd.DataFrame(file, columns=(['x', 'y']))
    return np.array(dframe)

if __name__ == '__main__':
    data = readDataExcel()
    f_fitness = Clustering()
    npop = 8
    ndim = 2
    nCluster = 3
    maxloop = 4
    # centroids = np.array([[5, 5], [3, 1], [6, 6]])
    bound = [1, 10]

    obj = standart_GA(npop, ndim, nCluster, maxloop, fitness_function=f_fitness, bound=bound)
    obj.mainProgram()
    f_fitness.plottingScatter(obj.bestCentroid)
    eva = evaluation(data,obj.bestCentroid)
    eva.get_SSE()
    eva.get_silhouette_score()
    eva.get_davies_bouldin()
    print("===============")
    obj = GA_mean(data, npop, ndim, nCluster, maxloop, fitness_function=f_fitness, bound=bound)
    obj.mainProgram()