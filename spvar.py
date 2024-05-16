import dimod.utilities
import numpy as np
import qubovert
import dimod
from dataclasses import dataclass
import scipy.stats as ss

EPS = 0.001

def generate_sample(h : dict, J : dict, sample_size : int, anneal_duration : int = 1000) -> qubovert.sim.AnnealResults:
    (Q, _) = dimod.ising_to_qubo(h, J)
    return qubovert.sim.anneal_qubo(Q, num_anneals=sample_size, anneal_duration=anneal_duration)

@dataclass
class SPVAR_default_params:
    h : dict
    J : dict
    sample_size : int
    fixing_threshold : int
    elite_threshold: int
    anneal_duration: int = 1000

@dataclass
class SPVAR_test_honest_params:
    h : dict
    J : dict
    total_num_anneals: int
    SPVAR_num_anneals : int
    fixing_threshold : int
    elite_threshold: int

def from_honest_to_default(params: SPVAR_test_honest_params) -> SPVAR_default_params:
    return SPVAR_default_params(
        params.h.copy(),
        params.J.copy(),
        params.SPVAR_num_anneals,
        params.fixing_threshold,
        params.elite_threshold
    )

class SPVAR:

    def __init__(self):
        pass

    # returns new h, J, mapping dict and offset
    def spvar(self, params : SPVAR_default_params) -> tuple[dict, dict, dict, int]:
        
        n = len(params.h.keys())

        samples = generate_sample(params.h, params.J, params.sample_size, params.anneal_duration)
        samples.sort()
        samples = samples[:int(params.sample_size * params.elite_threshold)]

        appears = np.array([np.array([sample.state[i] * 2 - 1 for sample in samples]) for i in range(n)])
        deviation = ss.tstd(appears, axis=1)
        avg = np.average(appears, axis=1)

        fix = deviation < params.fixing_threshold + EPS
        fixed = dict()

        for i in range(n):
            if (fix[i]):
                fixed[i] = 1 if avg[i] > 0 else -1

        offset = 0

        J = params.J.copy()
        h = params.h.copy()

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
        return tuple([h, J, fixed, offset])
        


    # Compare result from annealer on given task with
    # result from annealer on simplified by SPVAR task 
    # Returns tuple (result before SPVAR, result after SPVAR)
    def test_once(
            self,
            spvar_params: SPVAR_default_params,
            anneal_duration_one
        ) -> tuple[float, float, int]:

        solution_old = generate_sample(spvar_params.h, spvar_params.J, 1, anneal_duration_one).best

        (spvar_h, spvar_J, fixed, _) = self.spvar(spvar_params)

        solutionNew = generate_sample(spvar_h, spvar_J, 1, anneal_duration_one).best

        state_new = solutionNew.state

        cnt_fixed = len(fixed.keys())

        for v in fixed:
            state_new[v] = 0 if fixed[v] == -1 else 1

        before = solution_old.value
        (Q_new, _) = dimod.ising_to_qubo(h, J)
        after = qubovert.utils.QUBOMatrix(Q_new).value(state_new)
        return (before, after, cnt_fixed)

    # Comapares two solutions:
    # 1) best of params.total_num_anneals results from annealer on given task
    # 2) best of (params.total_num_anneals - params.SPVAR_num_anneals) results from annealer on simplified by SPVAR task
    # In last case we spend params.SPVAR_num_anneals anneals to run SPVAR
    # Returns tuple (result without spvar, result with spvar, count of fixed vars)
    def test_honest(self, params : SPVAR_test_honest_params) -> tuple[int, int, int]:
        solution_no_spvar = generate_sample(params.h, params.J, params.total_num_anneals).best
        no_spvar_result = solution_no_spvar.value

        default_params = from_honest_to_default(params)
        [h_spvar, J_spvar, fixed, _] = self.spvar(default_params)

        cnt_fixed = len(fixed.keys())

        num_anneals_after_spvar = params.total_num_anneals - params.SPVAR_num_anneals
        solution_spvar = generate_sample(h_spvar, J_spvar, num_anneals_after_spvar).best
        state_spvar = solution_spvar.state

        for v in fixed:
            state_spvar[v] = 0 if fixed[v] == -1 else 1

        # state_spvar = solution_spvar.state
        (Q_spvar, _) = dimod.ising_to_qubo(params.h, params.J)
        spvar_result = qubovert.utils.QUBOMatrix(Q_spvar).value(state_spvar)

        return tuple([no_spvar_result, spvar_result, cnt_fixed])