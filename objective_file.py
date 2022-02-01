import math
from Dataset import data

class Clustering:
    data = None
    def __init__(self):
        self.data = data()

    def fitness(self, x):
        return self.data.fitness(x)

    def plottingScatter(self, centroid):
        self.data.scatterDataCentroid(centroid)