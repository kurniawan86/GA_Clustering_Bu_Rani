import random

class aritmatik:
    male = []
    female = []

    def __init__(self, male, female):
        self.male = male
        self.female = female

    def getOffspring(self):
        c1 = random.random()
        c2 = 1 - c1
        child1 = []
        child2 = []
        for i in range(len(self.male)):
            child1.append(c1*self.male[i]+c2*self.female[i])
            child2.append(c2*self.male[i]+c1*self.female[i])
        return [child1, child2]