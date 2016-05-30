import random

import numpy as np


class GA:
    def __init__(self, n, mutation0, mutation1, func, len):
        self.n = n
        self.mutation0 = mutation0
        self.mutation1 = mutation1
        self.population = []
        self.func = func
        self.len = len

    def generate_population(self):
        for i in range(self.n):
            self.population.append(np.random.randint(2, size=self.len))

    def selection0(self, t, m):
        winners = []
        for i in range(m):
            contestants = random.sample(self.population, t)
            min = float('inf')
            fittest = []
            for contestant in contestants:
                if self.func(contestant) < min:
                    min = self.func(contestant)
                    fittest = contestant
            winners.append(fittest)
        while (len(winners)) > 0:
            i0 = random.randrange(0, len(winners))
            s0 = winners.pop(i0)
            i1 = random.randrange(0, len(winners))
            s1 = winners.pop(i1)
            child = self.cross(s0, s1)
            if random.randint(0, 100) < self.mutation0:
                for k in range(len(child)):
                    if random.randint(0, 100) < self.mutation1:
                        if child[k] == 0:
                            child[k] = 1
                        else:
                            child[k] = 0
            self.population.append(child)

    def selection1(self, k0, k1, t1):
        list_of_best = []
        for i in range(len(self.population)):
            list_of_best.append((self.func(self.population[i]), self.population[i]))
        list_of_best.sort(key=lambda x: x[0])
        new_population = []
        for i0 in range(k0):
            new_population.append(list_of_best[i0][1])
        for i1 in range(len(list_of_best) - 1, k1 - 1, -1):
            new_population.append(list_of_best[i1][1])
        for j in range(len(list_of_best) - k0 - k1):
            contestants = random.sample(self.population, t1)
            min = float('inf')
            fittest = []
            for contestant in contestants:
                if contestant[0] < min:
                    min = contestant[0]
                    fittest = contestant[1]
            new_population.append(fittest)
        return list_of_best[0]

    def cross(self, s0, s1):
        mask = np.random.randint(2, size=self.len)
        child = []
        for i in range(len(mask)):
            if mask[i] == 0:
                child.append(s0[i])
            else:
                child.append(s1[i])
        return np.array(child)

    def find_min(self, t, m0, k0, k1, t1):
        self.generate_population()
        end_criteria = True
        counter = 0
        form_best = None
        while end_criteria:
            self.selection0(t, m0)
            best = self.selection1(k0, k1, t1)
            if form_best is not None:
                if np.linalg.norm(form_best[0] - best[0]) < 0.001:
                    counter += 1
                form_best = best
            else:
                form_best = best
            if counter >= 100:
                end_criteria = False
        return form_best
