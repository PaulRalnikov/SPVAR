import numpy as np
import pandas as pd
import qubovert
from random import randint
import dimod
import sys
import io_maxcut

EPS = 0.001

def generate_sample(Q, sample_size : int) -> np.ndarray:
    return qubovert.sim.anneal_qubo(Q, num_anneals=sample_size, anneal_duration=20)

def spvar(h,
          J, 
          sample_size : int,
          fixing_threshold : float,
          elite_threshold : float
          ) -> np.ndarray:
    
    n = len(h.keys())
    Q = dimod.ising_to_qubo(h, J)[0]

    samples = generate_sample(Q, sample_size)
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
    return (h, J, fixed, offset)
    
def read_ising_from_file(path : str) -> tuple[dict, dict, int]:
    g = io_maxcut.read_gset_graph(path)
    Q_mat = io_maxcut.graph_to_qubo(g)
    shape = Q_mat.shape

    Q_dict = dict()    
    for i in range(shape[0]):
        for j in range(shape[1]):
            Q_dict[(i, j)] = Q_mat[i][j]

    return dimod.qubo_to_ising(Q_dict)

# returns tuple (result before SPVAR, result after SPVAR)
def test_spvar(h, J, sample_size, fixing_threshold, elite_threshold) -> tuple[float, float]:
    (Q, _) = dimod.ising_to_qubo(h, J)

    solution_old = generate_sample(Q, 1).best

    (spvar_h, spvar_J, fixed, _) = spvar(h, J, sample_size, fixing_threshold, elite_threshold)

    (spvar_Q, _) = dimod.ising_to_qubo(spvar_h, spvar_J)
    solutionNew = generate_sample(spvar_Q, 1).best

    state_new = solutionNew.state

    for v in fixed:
        state_new[v] = 0 if fixed[v] == -1 else 1

    before = solution_old.value
    after = qubovert.utils.QUBOMatrix(Q).value(state_new)
    return (before, after)


sample_size = 1_000
fixing_threshold = 1.0
elite_threshold = 0.2

testPathes = io_maxcut.files_in_directory("Gset")
testPathes.sort()

best_known_energy = {
    'G1.dat': 11624,
    'G3.dat': 11622,
    'G7.dat': 2006,
    'G11.dat': 564,
    'G15.dat': 3050,
    'G20.dat': 941,
    'G25.dat': 13340,
    'G28.dat': 3298,
    'G33.dat': 1382,
    'G40.dat': 2400,
    'G43.dat': 6660,
    'G44.dat': 6650
}

results = []
for testPath in testPathes:
    print(f"start test {testPath}")
    path = "Gset/" + testPath
    (h, J, offsetIsing) = read_ising_from_file(path)
    best_known = best_known_energy[testPath]
    (before, after) = test_spvar(h, J, sample_size, fixing_threshold, elite_threshold)

    before = -before
    after = -after

    results.append([before, after, best_known, str(round((after - before) / best_known * 100, 1)) + '%'])
    print(f"test {testPath} finished!")


print(
    pd.DataFrame(
        data=results,
        columns = ["result before SPVAR", "result after SPVAR", "best known", "percent"],
        index = testPathes
        )
)


    