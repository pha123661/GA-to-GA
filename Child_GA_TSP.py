import numpy as np
np.random.seed(0)


def Child_GA(Pc=0.5, Pm=0.01, NUM_CHROME=20, Optimal_sol=26, TSP_graph=[[0, 12, 1, 100],
                                                                        [12, 0,
                                                                         2, 3],
                                                                        [1, 2,
                                                                         0, 10],
                                                                        [100, 3, 10, 0]]):
    NUM_BIT = len(TSP_graph) - 1
    NUM_PARENT = NUM_CHROME
    NUM_CROSSOVER = max(int(Pc * NUM_CHROME / 2), 1)
    NUM_CROSSOVER_2 = NUM_CROSSOVER*2
    NUM_MUTATION = int(Pm * NUM_CHROME * NUM_BIT)
    MAX_NUM_ITERATION = 3000

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

    def mutation(p):
        for _ in range(NUM_MUTATION):
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
    best_so_far = 2**64

    for i in range(1, MAX_NUM_ITERATION+1):
        parent = selection(pop, pop_fit)
        offspring = crossover_uniform(parent)
        mutation(offspring)
        offspring_fit = evaluatePop(offspring)
        pop, pop_fit = replace(pop, pop_fit, offspring, offspring_fit)
        best_current_iteration = -np.max(pop_fit)

    return 2*MAX_NUM_ITERATION


if __name__ == "__main__":
    in_ = np.random.randint(1, 100, size=(20, 20))
    print(Child_GA(NUM_CHROME=30, TSP_graph=in_))
