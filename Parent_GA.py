import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp
import random
from Child_GA_TSP import Child_GA
np.random.seed(0)
random.seed(0)


class Parent_GA():
    def __init__(self, Manager, NUM_ITERATION=10, NUM_CHROME=20, NUM_BIT=19, Pc=0.5, Pm=0.01, PENALTY=1024):
        # Field of chromosome
        # [0:6]: self.Pc, [7:13]: self.Pm*100, [14:18]: self.NUM_CHROME,
        self.NUM_ITERATION = NUM_ITERATION
        self.NUM_CHROME = NUM_CHROME
        self.NUM_BIT = NUM_BIT
        self.PENALTY = PENALTY

        self.NUM_PARENT = NUM_CHROME
        self.NUM_CROSSOVER = int(Pc * NUM_CHROME / 2)
        self.NUM_MUTATION = int(Pm * NUM_CHROME * NUM_BIT)

        self.cache = Manager.dict()
        self.TSP_maps = [
            tuple(tuple(x)
                  for x in np.random.randint(1, 100, size=(14, 14)).tolist()),
            tuple(tuple(x)
                  for x in np.random.randint(200, 301, size=(14, 14)).tolist())
        ]

    def __initPop(self):
        return np.random.randint(2, size=(self.NUM_CHROME, self.NUM_BIT))

    def __Decode(self, x):
        def BitList2Int(BitList):
            out = 0
            for bit in BitList:
                out = (out << 1) | bit
            return out
        Pc = x[0:6]
        Pm = x[7:13]
        chrome = x[14:18]

        Pc = BitList2Int(Pc)
        Pc = 0.6 + (1-0.6) * (Pc/128)
        Pm = BitList2Int(Pm)
        Pm = 0 + (0.01-0) * (Pm/128)
        chrome = BitList2Int(chrome)
        chrome = int(20 + (30-20) * (chrome/32))

        return Pc, Pm, chrome

    def fitFunc(self, x):
        map = random.choice(self.TSP_maps)
        Pc, Pm, chrome = self.__Decode(x)
        key = (Pc, Pm, chrome, map)
        if key in self.cache:
            return self.cache[key]
        Num_penalty = 0
        if Pc < 0.6 or Pc > 1:
            Num_penalty += 1
        if Pm > 0.01:
            Num_penalty += 1
        if chrome > 30 or chrome < 20:
            Num_penalty += 1
        fit_value = 1 / \
            float(Child_GA(Pc=Pc, Pm=Pm, NUM_CHROME=chrome, TSP_graph=map) +
                  Num_penalty*self.PENALTY)
        self.cache[key] = fit_value
        return fit_value

    def __evaluatePop(self, P):
        with mp.Pool(mp.cpu_count()) as pool:
            return pool.map(self.fitFunc, P)

    def __selection(self, p, p_fit):
        a = []
        for _ in range(self.NUM_PARENT):
            [j, k] = np.random.choice(self.NUM_CHROME, 2, replace=False)
            if p_fit[j] > p_fit[k]:
                a.append(p[j])
            else:
                a.append(p[k])
        return a

    def __crossover(self, p):
        a = []
        for _ in range(self.NUM_CROSSOVER):
            c = np.random.randint(1, self.NUM_BIT)
            [j, k] = np.random.choice(self.NUM_PARENT, 2, replace=False)
            a.append(np.concatenate(
                (p[j][0: c], p[k][c: self.NUM_BIT]), axis=0))
            a.append(np.concatenate(
                (p[k][0: c], p[j][c: self.NUM_BIT]), axis=0))
        return a

    def __mutation(self, p):
        for _ in range(self.NUM_MUTATION):
            row = np.random.randint(self.NUM_CROSSOVER * 2)
            col = np.random.randint(self.NUM_BIT)
            p[row][col] = (p[row][col] + 1) % 2

    def __sortChrome(self, a, a_fit):
        a_index = range(len(a))
        a_fit, a_index = zip(*sorted(zip(a_fit, a_index), reverse=True))
        return [a[i] for i in a_index], a_fit

    def __replace(self, p, p_fit, a, a_fit):
        b = np.concatenate((p, a), axis=0)
        b_fit = p_fit + a_fit
        b, b_fit = self.__sortChrome(b, b_fit)
        return b[:self.NUM_CHROME], list(b_fit[:self.NUM_CHROME])

    def Eval(self, plot=False, return_fit=False):
        pop = self.__initPop()
        pop_fit = self.__evaluatePop(pop)
        mean_outputs = [np.average([int(1/p) for p in pop_fit])]
        best_outputs = [np.max([int(1/p) for p in pop_fit])]
        for _ in range(self.NUM_ITERATION):
            parent = self.__selection(pop, pop_fit)
            offspring = self.__crossover(parent)
            self.__mutation(offspring)
            offspring_fit = self.__evaluatePop(offspring)
            pop, pop_fit = self.__replace(
                pop, pop_fit, offspring, offspring_fit)
            mean_outputs.append(np.average([int(1/p) for p in pop_fit]))
            best_outputs.append(np.max([int(1/p) for p in pop_fit]))
            print([int(1/p) for p in pop_fit])
        if plot:
            outputs = [mean_outputs]
            self.plot(outputs)
        if return_fit:
            pop_fit[0]

    def plot(self, outputs):
        for output in outputs:
            plt.plot(output)
        plt.xlabel("Iteration")
        plt.ylabel("Fitness")
        plt.show()


if __name__ == "__main__":
    a = Parent_GA(Manager=mp.Manager())
    a.Eval(plot=True)
