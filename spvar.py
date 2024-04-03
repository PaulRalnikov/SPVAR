import numpy as np
import pandas as pd
import qubovert
import dimod
import io_maxcut
import sys

EPS = 0.001

def generate_sample(Q, sample_size : int, anneal_duration) -> np.ndarray:
    return qubovert.sim.anneal_qubo(Q, num_anneals=sample_size, anneal_duration=anneal_duration)

def spvar(h,
          J, 
          sample_size : int,
          fixing_threshold : float,
          elite_threshold : float,
          anneal_duration
          ) -> np.ndarray:
    
    n = len(h.keys())
    Q = dimod.ising_to_qubo(h, J)[0]

    samples = generate_sample(Q, sample_size, anneal_duration)
    samples.sort()
    # samples.reverse()
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
def test_spvar(h,
               J,
               sample_size,
               fixing_threshold,
               elite_threshold,
               anneal_duration_one,
               anneal_duration_many) -> tuple[float, float]:
    (Q, _) = dimod.ising_to_qubo(h, J)

    solution_old = generate_sample(Q, 1, anneal_duration_one).best

    (spvar_h, spvar_J, fixed, _) = spvar(
        h,
        J,
        sample_size,
        fixing_threshold,
        elite_threshold,
        anneal_duration_many
    )

    (spvar_Q, _) = dimod.ising_to_qubo(spvar_h, spvar_J)
    solutionNew = generate_sample(spvar_Q, 1, anneal_duration_one).best

    state_new = solutionNew.state

    for v in fixed:
        state_new[v] = 0 if fixed[v] == -1 else 1

    before = solution_old.value
    after = qubovert.utils.QUBOMatrix(Q).value(state_new)
    return (before, after)

def test_criteria(sample_size,
                  fixing_threshold,
                  elite_threshold,
                  anneal_duration_many,
                  anneal_duration_one) :
    
    print("start testing criterea:")
    print(f"sample_size: {sample_size}")
    print(f"fixing_threshold: {fixing_threshold}")
    print(f"elite_threshold: {elite_threshold}")
    print(f"anneal_duration_one: {anneal_duration_one}")
    print(f"anneal_duration_many: {anneal_duration_many}")

    testPathes = io_maxcut.files_in_directory("Gset")

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
        (h, J, _) = read_ising_from_file(path)
        best_known = best_known_energy[testPath]
        (before, after) = test_spvar(
            h,
            J,
            sample_size,
            fixing_threshold,
            elite_threshold,
            anneal_duration_one,
            anneal_duration_many
        )

        before = -before
        after = -after

        results.append([
            before,
            after,
            best_known,
            str(round((before - best_known) / best_known * 100, 1)) + '%',
            str(round((after - best_known) / best_known * 100, 1)) + '%',  
            str(round((after - before) / before * 100, 1)) + '%',  
        ])
        print(f"test {testPath} finished!")

    f = open("output.txt", "a")
    
    f.write(f"sample_size: {sample_size}\n")
    f.write(f"fixing_threshold: {fixing_threshold}\n")
    f.write(f"elite_threshold: {elite_threshold}\n")
    f.write(f"anneal_duration_one: {anneal_duration_one}\n")
    f.write(f"anneal_duration_many: {anneal_duration_many}\n")
    f.write(
        pd.DataFrame(
            data=results,
            columns = [
                "result before SPVAR",
                "result after SPVAR",
                "best known",
                "(before - best) / best",
                "(after - best) / best",
                "(after - before) / before"
            ],
            index = testPathes
            ).to_string() + "\n",
    )
    f.write("===================================================================================================================================\n")
    f.close()
    

test_criteria(1000, 1.0, 0.1, 1000, 1000)
test_criteria(1000, 1.0, 0.2, 1000, 1000)
test_criteria(1000, 1.0, 0.3, 1000, 1000)
test_criteria(1000, 1.0, 0.4, 1000, 1000)
test_criteria(1000, 1.0, 0.5, 1000, 1000)
test_criteria(1000, 1.0, 0.6, 1000, 1000)
test_criteria(1000, 1.0, 0.7, 1000, 1000)
test_criteria(1000, 1.0, 0.8, 1000, 1000)
test_criteria(1000, 1.0, 0.9, 1000, 1000)
test_criteria(1000, 1.0, 1.0, 1000, 1000)
    