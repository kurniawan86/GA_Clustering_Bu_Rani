from objective_file import Clustering
import numpy as np
import pandas as pd
from evaluation_file import evaluation
from GA_file import *
import warnings
from GAPoly_file import *
from statistics import mean

def readDataExcel():
    file = pd.read_excel(open('borneo.xlsx', 'rb'))
    dframe = pd.DataFrame(file, columns=(['x', 'y']))
    return np.array(dframe)

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    data = readDataExcel()
    # print(data)
    f_fitness = Clustering()
    npop = 10
    ndim = 2
    nCluster = 4
    maxloop = 50
    # centroids = np.array([[5, 5], [3, 1], [6, 6]])
    bound = [0, 1]
    
    # hasil = []
    # sse = []
    # silhouutte = []
    # davies = []
    # loop = []
    # for i in range(10):
    #     # obj = standart_GA(npop, ndim, nCluster, maxloop, fitness_function=f_fitness, bound=bound)
    #     obj = standart_GA(npop, ndim, nCluster, maxloop, fitness_function=f_fitness)
    #     obj.mainProgram()
    #     f_fitness.plottingScatter(obj.bestCentroid)
    #     eva = evaluation(data,obj.bestCentroid)
    #     # eva.get_SSE()
    #     # eva.get_silhouette_score()
    #     # eva.get_davies_bouldin()
    #     sse.append(eva.get_SSE())
    #     silhouutte.append(eva.get_silhouette_score())
    #     davies.append(eva.get_davies_bouldin())
    #     loop.append(obj.iterstop)
    #
    #     # hasil.append(sse)
    #     # hasil.append(silhouutte)
    #     # hasil.append(davies)
    #     # hasil.append(loop)
    # # hasil =np.array(hasil)
    # print("SSE ", sse)
    # print("Sihloute ", silhouutte)
    # print("davies ", davies)
    # print("iterasi stop ", loop)
    #
    # print("avg SSE ", mean(sse))
    # print("avg Sihloute ", mean(silhouutte))
    # print("avg davies ", mean(davies))
    # print("avg iterasi stop ", mean(loop))

    # GA MEAN
    print("===============")
    hasil = []
    sse = []
    silhouutte = []
    davies = []
    loop = []
    for i in range(2):
        print("RUN PROGRAM KE:", i+1)
        obj1 = GA_mean(data, npop, ndim, nCluster, maxloop, fitness_function=f_fitness)
        obj1.mainProgram()
        f_fitness.plottingScatter(obj1.bestCentroid)
        eva1 = evaluation(data, obj1.bestCentroid)
        eva1.get_SSE()
        eva1.get_silhouette_score()
        eva1.get_davies_bouldin()
        print("LLOP :",obj1.iterstop)
        loop.append(obj1.iterstop)

        sse.append(eva1.get_SSE())
        silhouutte.append(eva1.get_silhouette_score())
        davies.append(eva1.get_davies_bouldin())
        loop.append(obj1.iterstop)

        hasil.append(sse)
        hasil.append(silhouutte)
        hasil.append(davies)
        hasil.append(loop)

    print("SSE ", sse)
    print("Sihloute ", silhouutte)
    print("davies ", davies)
    print("iterasi stop ", loop)

    print("avg SSE ", mean(sse))
    print("avg Sihloute ", mean(silhouutte))
    print("avg davies ", mean(davies))
    print("avg iterasi stop ", mean(loop))
    
    # GA polygamy
    # mateHasil = []
    # mateHAsilAvg = []
    # fitnessAll = []
    # for mate in range(2):
    #     hasil = []
    #     hasilAvg = []
    #     sse = []
    #     silhouutte = []
    #     davies = []
    #     loop = []
    #     for i in range(3):
    #         nmate = 2+mate
    #         print("===============")
    #         print("n MATE :",nmate)
    #         obj3 = GA_poly(data, npop, ndim, nCluster, maxloop, nmate, fitness_function=f_fitness)
    #         obj3.mainProgram()
    #         fitnessAll.append(obj3.fitnessAll)
    #         loop.append(obj3.stoploop())
    #         # f_fitness.plottingScatter(obj3.bestCentroid)
    #         eva3 = evaluation(data, obj3.bestCentroid)
    #         sse.append(eva3.get_SSE())
    #         silhouutte.append(eva3.get_silhouette_score())
    #         davies.append(eva3.get_davies_bouldin())
    #         f_fitness.plottingScatter(obj3.bestCentroid)
    #     sse_average = mean(sse)
    #     sil_average = mean(silhouutte)
    #     dav_average = mean(davies)
    #     iterasi_average = mean(loop)
    #     hasil.append(sse)
    #     hasil.append(silhouutte)
    #     hasil.append(davies)
    #     hasil.append(loop)
    #     hasilAvg.append(sse_average)
    #     hasilAvg.append(sil_average)
    #     hasilAvg.append(dav_average)
    #     hasilAvg.append(iterasi_average)
    #
    #     # print(np.array(hasil))
    #     mateHasil.append(hasil)
    #     mateHAsilAvg.append(hasilAvg)
    # print("mate Hasil\n", np.array(mateHasil))
    # print("mate Hasil AVG\n", np.array(mateHAsilAvg))
    
    # print("===============")
    # obj2 = GA_poly_mean(data, npop, ndim, nCluster, maxloop, fitness_function=f_fitness)
    # obj2.mainProgram()
    # # f_fitness.plottingScatter(obj2.bestCentroid)
    # eva2 = evaluation(data, obj2.bestCentroid)
    # eva2.get_SSE()
    # eva2.get_silhouette_score()
    # eva2.get_davies_bouldin()