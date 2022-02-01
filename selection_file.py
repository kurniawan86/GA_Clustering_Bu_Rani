import random
import numpy as np

class rw:
    fit = []
    chrom = []
    def __init__(self, chromosom, fitness):
        self.fit = fitness
        self.chrom = chromosom

    def __cal_perFit(self):
        FIT = []
        for i in range(len(self.fit)):
            FIT.append(1/(1+self.fit[i]))
        return FIT

    def __cal_fn(self):
        fn = []
        FIT = self.__cal_perFit()
        val_min = min(FIT)
        for i in range(len(self.fit)):
            fn.append(FIT[i]-val_min)
        return fn

    def __cal_fkn(self):
        fkn = []
        fn = self.__cal_fn()
        fkn.append(fn[0])
        for i in range(1,len(self.fit)):
            fkn.append(fkn[i-1]+fn[i])
        return fkn

    def __cal_nfkn(self):
        fkn = self.__cal_fkn()
        maksi = max(fkn)
        nfkn = []
        for i in range(len(self.fit)):
            if maksi == min(fkn):
                nfkn.append(i)
            else:
                nfkn.append(fkn[i] / maksi)
        return nfkn

    def __pickParent(self):
        rw = self.__cal_nfkn()
        rand = random.random()
        for i in range(len(self.fit)):
            if rand < rw[i]:
                index = i
                break
        return index

    def pick2Parents(self):
        male = self.__pickParent()
        female = self.__pickParent()
        return [male,female]

class turnament:
    pass