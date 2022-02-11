from objective_file import Clustering
import numpy as np
import pandas as pd
from evaluation_file import evaluation
from GA_file import *
import warnings

def readDataExcel():
    file = pd.read_excel(open('borneo.xlsx', 'rb'))
    dframe = pd.DataFrame(file, columns=(['x', 'y']))
    return np.array(dframe)

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    data = readDataExcel()
    # print(data)
    f_fitness = Clustering()
    npop = 20
    ndim = 2
    nCluster = 3
    maxloop = 50
    # centroids = np.array([[5, 5], [3, 1], [6, 6]])
    bound = [0, 1]

    # obj = standart_GA(npop, ndim, nCluster, maxloop, fitness_function=f_fitness, bound=bound)
    obj = standart_GA(npop, ndim, nCluster, maxloop, fitness_function=f_fitness)
    obj.mainProgram()
    # f_fitness.plottingScatter(obj.bestCentroid)
    eva = evaluation(data,obj.bestCentroid)
    eva.get_SSE()
    eva.get_silhouette_score()
    eva.get_davies_bouldin()
    print("===============")
    obj1 = GA_mean(data, npop, ndim, nCluster, maxloop, fitness_function=f_fitness)
    obj1.mainProgram()
    f_fitness.plottingScatter(obj1.bestCentroid)
    eva1 = evaluation(data, obj1.bestCentroid)
    eva1.get_SSE()
    eva1.get_silhouette_score()
    eva1.get_davies_bouldin()
