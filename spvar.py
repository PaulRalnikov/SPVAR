import numpy as np
import pandas as pd
import qubovert
from random import randint
import dimod

EPS = 0.001

def generateSample(Q, sample_size : int) -> np.ndarray:
    return qubovert.sim.anneal_qubo(Q, num_anneals=sample_size, anneal_duration=30)

def spvar(J,
          h, 
          sample_size : int,
          fixing_threshold : float,
          elite_threshold : float
          ) -> np.ndarray:
    
    n = len(h.keys())
    Q = dimod.ising_to_qubo(h, J)[0]

    samples = generateSample(Q, sample_size)
    samples.sort()
    samples.reverse()
    samples = samples[:int(sample_size * elite_threshold)]

    new_sample_size = len(samples)

    sum_appear = np.full(n, 0)
    for sample in samples:
        for i in range (n):
            sum_appear[i] += 2 * sample.state[i] - 1
    
    fix = np.array([(abs(x) / new_sample_size - fixing_threshold > -EPS) for x in np.nditer(sum_appear)])
    fixed = dict()

    for i in range(n):
        if (fix[i]):
            fixed[i] = 1 if sum_appear[i] > 0 else -1

    offset = 0

    for i in range(n):
        if (fix[i]):
            offset += J[(i, i)]
            J.pop((i, i))
        
        for j in range(i + 1, n):
            if not fix[i] and not fix[j]:
                continue

            sum_J = J[(i, j)] + J[(j, i)]

            if fix[i] and fix[j]:
                offset += sum_J * fixed[i] * fixed[j]
                
            if not fix[i]:
                h[i] += sum_J * fixed[j]
            if not fix[j]:
                h[j] += sum_J * fixed[i]

            J.pop((i, j))
            J.pop((j, i))
    
    for i in range(n):
        if fix[i]:
            h.pop(i)
    return (J, h, fixed, offset)
    
count_tests = 10
results = []
for _ in range(count_tests):
    n = 100
    J = dict()
    h = dict()
    for i in range(n):
        h[i] = randint(0, 100)
        for j in range(n):
            J[(i, j)] = randint(0, 100)

    (Q, off_Q) = dimod.ising_to_qubo(h, J)

    solution_old = generateSample(Q, 1).best

    (new_J, new_h, fixed, offset) = spvar(J, h, 10000, 1.0, 0.8)

    (new_Q, off_new_Q) = dimod.ising_to_qubo(new_h, new_J)
    solution_new = generateSample(new_Q, 1).best

    state_new = solution_new.state

    for v in fixed:
        state_new[v] = 0 if fixed[v] == -1 else 1

    before = solution_old.value
    after = qubovert.utils.QUBOMatrix(Q).value(state_new)

    results.append([before, after, (before - after) / before * 100])


print(results)
print(pd.DataFrame(results, columns = ["result before SPVAR", "result after SPVAR", "percent"]))