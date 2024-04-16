import numpy as np
import qubovert
import dimod
from math import sqrt

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
    (Q, _) = dimod.ising_to_qubo(h, J)

    samples = generate_sample(Q, sample_size, anneal_duration)
    samples.sort()
    samples = samples[:int(sample_size * elite_threshold)]

    new_sample_size = len(samples)

    sum_appear = np.full(n, 0)
    for i in range (n):
        for sample in samples:
            sum_appear[i] += 2 * sample.state[i] - 1

    middle = [x / new_sample_size for x in np.nditer(sum_appear)]

    deviation = np.full(n, 0)
    for i in range (n):
        for sample in samples:
            val = (2 * sample.state[i] - 1 - middle[i])
            deviation[i] += val * val

    fix = np.array([(sqrt(x / new_sample_size) < fixing_threshold + EPS) for x in np.nditer(deviation)])
    fixed = dict()

    for i in range(n):
        if (fix[i]):
            fixed[i] = 1 if sum_appear[i] > 0 else -1

    offset = 0

    for i in range(n):
        if (fix[i]):
            if ((i, i) in J):
                offset += J[(i, i)]
                J.pop((i, i))
        
        for j in range(i + 1, n):
            if not fix[i] and not fix[j]:
                continue

            sum_J = J.get((i, j), 0) + J.get((j, i), 0)

            if fix[i] and fix[j]:
                offset += sum_J * fixed[i] * fixed[j]
                
            if not fix[i]:
                h[i] += sum_J * fixed[j]
            if not fix[j]:
                h[j] += sum_J * fixed[i]

            if ((i, j) in J):
                J.pop((i, j))
            if (j, i) in J:
                J.pop((j, i))
    
    for i in range(n):
        if fix[i] and i in h:
            h.pop(i)
    return (h, J, fixed, offset)
    


# returns tuple (result before SPVAR, result after SPVAR)
def test_spvar(h,
               J,
               sample_size,
               fixing_threshold,
               elite_threshold,
               anneal_duration_one,
               anneal_duration_many) -> tuple[float, float, int]:
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

    cnt_fixed = len(fixed.keys())

    for v in fixed:
        state_new[v] = 0 if fixed[v] == -1 else 1

    before = solution_old.value
    after = qubovert.utils.QUBOMatrix(Q).value(state_new)
    return (before, after, cnt_fixed)

