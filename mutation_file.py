import random

class RandomUniform:
    dim = 0
    def __init__(self, ndim):
        self.dim = ndim

    def getOffsping(self):
        n = self.dim
        child = []
        for i in range(n):
            child.append(random.random())
        return child

    def getOffspings(self,k):
        child = []
        for i in range(k):
            child.append(self.__creteIndividu())
        return child