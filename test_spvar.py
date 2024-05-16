from dataclasses import dataclass
import spvar
import io_maxcut
import dimod
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functools import *
import os
from math import *

# Returns tuple[h, J, offset]
def read_ising_from_Gset_file(path : str) -> tuple[dict, dict, int]:
    g = io_maxcut.read_gset_graph(path)
    Q_mat = io_maxcut.graph_to_qubo(g)
    shape = Q_mat.shape

    Q_dict = dict()    
    for i in range(shape[0]):
        for j in range(shape[1]):
            Q_dict[(i, j)] = Q_mat[i][j]

    return dimod.qubo_to_ising(Q_dict)

#returns Q
def read_qubo_from_csv_file(path : str) -> dict:
    f = open(path, "r")
    i = 0
    Q = dict()
    for line in f.readlines():
        coefs = list(map(float, line.split(',')))
        for j in range(len(coefs)):
            Q[(i, j)] = coefs[j]
        i += 1
    return Q

#Returns list of tuples [h, J, offset]
def read_qubo_matrices(dir_name : str) -> list[tuple[str, dict, dict]]:
    testPathes = io_maxcut.files_in_directory(dir_name)

    result = []
    for testPath in testPathes:
        path = f"{dir_name}/" + testPath
        Q = read_qubo_from_csv_file(path)
        [h, J, _] = dimod.qubo_to_ising(Q)
        result.append(tuple([testPath, h, J]))
    
    return result

@dataclass
class Test_honest_params:
    total_num_anneals : int
    SPVAR_num_anneals : int
    fixing_threshhold : float
    elite_threshold : float

# Does tests from params and returns results in list of DataFrames
# with columns [Result without SPVAR, Result with SPVAR, % fixed vars]
def test_honest(params : list[Test_honest_params], dir_tests : str) -> list[pd.DataFrame]:    
    tests = read_qubo_matrices(dir_tests)
    results = []
    names = [name for [name, _, _] in tests]
    s = spvar.SPVAR()
    for param in params:
        print(f"start test {param}")
        test_results = []
        for [_, h, J] in tests:
            num_vars = len(h.keys())
            spvar_params = spvar.SPVAR_honest_params(
                h,
                J,
                param.total_num_anneals,
                param.SPVAR_num_anneals,
                param.fixing_threshhold,
                param.elite_threshold
            )
            [no_spvar_result, spvar_result, cnt_fixed] = s.test_honest(spvar_params)
            test_results.append([no_spvar_result, spvar_result, round(cnt_fixed / num_vars * 100, 1)])

        columns = ["Result without SPVAR", "Result with SPVAR", "% fixed vars"]

        results.append(pd.DataFrame(test_results, index = names, columns=columns))
    
    return results

# Draws diagrams by results of honest tests and puts it
# to file results.pdf in directory dir_results
# typical case of using:
#     test_results = test_honest(...)
#     draw_bars_by_honest_results(..., test_results)
#
def draw_bars_by_honest_results(
        total_num_anneals : int,
        SPVAR_num_anneals_range : range,
        fixing_threshhold : float,
        elite_threshold : float,
        dir_results : str,
        results : list[pd.DataFrame]
        ):
    
    if len(results) == 0:
        print("No results:(")
        return

    test_names = results[0].index

    sz = len(test_names)
    
    plt.rcParams.update({'font.size': 21})
    fig, axs = plt.subplots(2, sz, figsize=(15 * sz, 25))

    plt.suptitle(f"params:\n" +
                 f"total_num_anneals = {total_num_anneals}\n" +
                 f"fixing_threshhold = {fixing_threshhold}\n" +
                 f"elite_threshold = {elite_threshold}")

    for i in range (len(test_names)):
        test_name = test_names[i]

        columns = ["Result without SPVAR", "Result with SPVAR", "% fixed vars"]

        data = {res : np.array([el.loc[test_name, res] for el in results]) for res in columns}
        data["SPVAR number anneals"] = list(SPVAR_num_anneals_range)
        frame = pd.DataFrame(data)

        x = frame["SPVAR number anneals"]
        y = (frame["Result with SPVAR"] - frame["Result without SPVAR"]) / np.abs(frame["Result without SPVAR"]) * 100

        ax = axs[0] if sz == 1 else axs[0, i]

        ax.bar(x, y, width=SPVAR_num_anneals_range.step / 5)
        ax.set_ylabel("Growth of target function in %")
        ax.set_xlabel("SPVAR number anneals")
        ax.set_title(test_name, pad=30)

        last = SPVAR_num_anneals_range.start
        for el in SPVAR_num_anneals_range:
            last = el

        ax.hlines(
            y = 0,
            xmin=SPVAR_num_anneals_range.start,
            xmax=last
        )
        
        for container in ax.containers:
            ax.bar_label(container, fmt='%.1f')


        ax = axs[1] if sz == 1 else axs[1, i]

        x = frame["SPVAR number anneals"]
        y = frame["% fixed vars"]

        ax.bar(x, y, width=SPVAR_num_anneals_range.step / 5)
        ax.set_ylabel("Percent of fixed vars")
        ax.set_xlabel("SPVAR number anneals")

        ax.hlines(
            y = 0,
            xmin=SPVAR_num_anneals_range.start,
            xmax=last
        )
        
        for container in ax.containers:
            ax.bar_label(container, fmt='%.1f')        


    plt.savefig(f"{dir_results}\\results.pdf")
    plt.close()

# Does some honest tests with params [total_num_anneals, SPVAR_num_anneals, fixing_threshhold, elite_threshold]
# and saves results to files with names test_{SPVAR_num_anneals}.csv in dir_results directory
# (SPVAR_num_anneals is an element of SPVAR_num_anneals_range);
# then draw bar charts by results and saves in to the dir_results directory.
# Also, if cached = True, this function gets results of tests from corresponding files in dir_results directory
# (in this case results of tests have to be placed in dir_results directory)
def test_different_num_anneals(
        total_num_anneals : int,
        SPVAR_num_anneals_range : range,
        fixing_threshhold : float,
        elite_threshold : float,
        dir_results : str,
        dir_data : str,
        cached : bool = False
        ):
    try:
        path = os.path.join(os.path.dirname(__file__), dir_results)
        os.mkdir(path)
    except OSError:
        pass
    
    results = []
    if cached:
        results = [
            pd.read_csv(f"{dir_results}\\test_{num}.csv", header=0, index_col=0)
            for num in SPVAR_num_anneals_range
        ]
    else:
        params = [
            Test_honest_params(
                total_num_anneals,
                SPVAR_num_anneals,
                fixing_threshhold,
                elite_threshold
            ) for SPVAR_num_anneals in SPVAR_num_anneals_range
        ]

        # test file exists
        for i in range(len(params)):
            fout = open(f"{dir_results}\\test_{params[i].SPVAR_num_anneals}.csv", 'a')
            fout.close()
        
        results = test_honest(params, dir_data)

        for i in range(len(params)):
            fout = open(f"{dir_results}\\test_{params[i].SPVAR_num_anneals}.csv", 'w')
            results[i].to_csv(fout)
            fout.close()

    draw_bars_by_honest_results(
        total_num_anneals,
        SPVAR_num_anneals_range,
        fixing_threshhold,
        elite_threshold,
        dir_results,
        results
    )    

def main():

    # sample of using:

    data_dir = "test_data\\qubo_matrices"
    results_dir = "test_results\\old"

    test_different_num_anneals(200, range(100, 200, 10), 0.0, 0.2,
                               f"{results_dir}\\test_200_0.0_0.2", data_dir, True)
    
    test_different_num_anneals(200, range(100, 200, 10), 0.1, 0.2,
                               f"{results_dir}\\test_200_0.1_0.2", data_dir, True)
    
    test_different_num_anneals(1000, range(100, 1000, 100), 0.0, 0.1,
                               f"{results_dir}\\test_1000_0.0_0.1", data_dir, True)
    
    test_different_num_anneals(1000, range(100, 1000, 100), 0.0, 0.2,
                               f"{results_dir}\\test_1000_0.0_0.2", data_dir, True)
    
    test_different_num_anneals(1000, range(100, 1000, 100), 0.1, 0.1,
                               f"{results_dir}\\test_1000_0.1_0.1", data_dir, True)

    test_different_num_anneals(1000, range(100, 1000, 100), 0.1, 0.2,
                               f"{results_dir}\\test_1000_0.1_0.2", data_dir, True)
    
    

main()