import numpy as np
import qubovert
from random import randint

def generateSample(Q : np.ndarray, sample_size : int) -> np.ndarray:
    return qubovert.sim.anneal_qubo(Q, num_anneals=sample_size, anneal_duration=5)
    for _ in range(sample_size):
        ans.append(qubovert.sim.anneal_qubo(Q, anneal_duration=5))
    return np.array(ans)

def spvar(Q, #QUBOMatrix object
          sample_size : int,
          fixing_threshold : float,
          elite_threshold : float
          ) -> np.ndarray:
    samples = generateSample(Q, sample_size)
    samples.sort()
    samples = samples[int(sample_size * (1 - elite_threshold)):]
    print(samples)
    
    n = Q.max_index + 1
    sum_appear = np.full(n, 0)

    for sample in samples:
        for i in range (n):
            sum_appear[i] += sample.state[i]
    print(sum_appear)

    return sample
    
n = 100
Q_matrix = dict()
Q = qubovert.utils.QUBOMatrix()
for i in range(n):
    for j in range(n):
        Q[(i, j)] = randint(-100, 100)
# print(Q)

print(spvar(Q, 10, 1.0, 0.5))
