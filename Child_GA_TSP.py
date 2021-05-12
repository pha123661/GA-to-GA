import numpy as np
import random


class MEMORY():
    def __init__(self, capacity) -> None:
        self.container = list()
        self.capacity = capacity

    def add(self, value):
        while len(self.container) >= self.capacity:
            self.container.pop(0)
        self.container.append(value)

    def get(self):
        return (sum(self.container) / len(self.container)) if len(self.container) else 0


def Child_GA(Pc, Pm, NUM_CHROME, TSP_graph):
    NUM_BIT = len(TSP_graph) - 1
    NUM_PARENT = NUM_CHROME
    NUM_CROSSOVER = max(int(Pc * NUM_CHROME / 2), 1)
    NUM_CROSSOVER_2 = NUM_CROSSOVER*2
    MAX_NUM_ITERATION = 30000

    def initPop():
        p = []
        for _ in range(NUM_CHROME):
            p.append(np.random.permutation(range(1, NUM_BIT+1)))
        return p

    def fitFunc(x):
        cost = TSP_graph[0][x[0]]
        for i in range(NUM_BIT-1):
            cost += TSP_graph[x[i]][x[i+1]]
        cost += TSP_graph[x[NUM_BIT-1]][0]
        return -cost

    def evaluatePop(P):
        return [fitFunc(p) for p in P]

    def selection(p, p_fit):
        a = []
        for _ in range(NUM_PARENT):
            [j, k] = np.random.choice(
                NUM_CHROME, 2, replace=False)
            if p_fit[j] > p_fit[k]:
                a.append(p[j].copy())
            else:
                a.append(p[k].copy())
        return a

    def crossover_uniform(p):
        '''
        TODO: cx2 crossover method
        '''
        a = []
        for _ in range(NUM_CROSSOVER):
            mask = np.random.randint(2, size=NUM_BIT)
            [j, k] = np.random.choice(
                NUM_PARENT, 2, replace=False)
            child1, child2 = p[j].copy(), p[k].copy()
            remain1, remain2 = list(p[j].copy()), list(p[k].copy())

            for m in range(NUM_BIT):
                if mask[m] == 1:
                    remain2.remove(child1[m])
                    remain1.remove(child2[m])
            t = 0
            for m in range(NUM_BIT):
                if mask[m] == 0:
                    child1[m] = remain2[t]
                    child2[m] = remain1[t]
                    t += 1
            a.append(child1)
            a.append(child2)
        return a

    def TF(name):
        # name = Pc or Pm
        if name == "Pc":
            return random.choices([True, False], weights=[Pc, 1-Pc])[0]
        elif name == "Pm":
            return random.choices([True, False], weights=[Pm, 1-Pm])[0]
        else:
            raise IndexError(name, "not found")

    def mutation(p):
        for _ in range(NUM_CHROME):
            if TF("Pm"):
                row = np.random.randint(NUM_CROSSOVER_2)
                [j, k] = np.random.choice(NUM_BIT, 2)
                p[row][j], p[row][k] = p[row][k], p[row][j]

    def sortChrome(a, a_fit):
        a_index = range(len(a))
        a_fit, a_index = zip(*sorted(zip(a_fit, a_index), reverse=True))
        return [a[i] for i in a_index], a_fit

    def replace(p, p_fit, a, a_fit):
        b = np.concatenate((p, a), axis=0)
        b_fit = p_fit + a_fit
        b, b_fit = sortChrome(b, b_fit)
        return b[:NUM_CHROME], list(b_fit[:NUM_CHROME])

    pop = initPop()
    pop_fit = evaluatePop(pop)
    memory = MEMORY(capacity=len(TSP_graph)**2)

    for i in range(1, MAX_NUM_ITERATION+1):
        parent = selection(pop, pop_fit)
        offspring = crossover_uniform(parent)
        mutation(offspring)
        offspring_fit = evaluatePop(offspring)
        pop, pop_fit = replace(pop, pop_fit, offspring, offspring_fit)
        mean = -1 * np.average(pop_fit)
        if abs(mean - memory.get()) == 0:
            return i
        memory.add(mean)

    return 2*MAX_NUM_ITERATION


if __name__ == "__main__":
    pass
