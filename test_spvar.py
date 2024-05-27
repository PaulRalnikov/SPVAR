from dataclasses import dataclass
import spvar
import read_matrices
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functools import *
import os
from math import *

DIR_OLD_RESULTS = "old_test_results"
DIR_RESULTS = "test_results"
DIR_DATA = "test_data"

# draws diagram by results (.csv files) in directory dir_results
# plot will be places in dir_results
def draw_plot(dir_results : str):
    files = []

    for file in os.listdir(dir_results):
        if (os.path.splitext(file)[1] == ".csv"):
            files.append(file)
    files.sort()
    sz = len(files)

    plt.rcParams.update({'font.size': 21})
    fig, axs = plt.subplots(2, sz, figsize=(15 * sz, 25))

    task_name = dir_results.split("\\")[-1]
    plt.suptitle(f"Task {task_name}")
    
    for i in range(sz):
        file = files[i]
        splitted = os.path.splitext(file)[0].split("_")

        total_num_anneals = int(splitted[0])
        fixing_thereshold = float(splitted[1])
        elite_thereshhold = float(splitted[2])

        file_path = dir_results + "\\" + file
        frame = pd.read_csv(file_path, header = 0, index_col = 0)
        index = frame["SPVAR num anneals"].to_list()
        
        start_range = index[0]
        last_range = index[-1]
        index_sz = len(index)
        step = 0 if sz == 0 else (last_range - start_range) / (index_sz - 1)

        x = index
        y = (frame["Result with SPVAR"] - frame["Result without SPVAR"]) / np.abs(frame["Result without SPVAR"]) * 100
        ax = axs[0] if sz == 1 else axs[0, i]
        width = step / 3

        ax.bar(x, y, width=width)
        ax.set_ylabel("Growth of target function in %")
        ax.set_xlabel("SPVAR number anneals")
        ax.set_title(f"params:\n"+
                     f"total_num_anneals = {total_num_anneals},\n"
                     f"fixing_thereshold = {fixing_thereshold},\n"
                     f"elite_thereshhold = {elite_thereshhold},",
                     pad=30)
        ax.hlines(
            y = 0,
            xmin=start_range,
            xmax=last_range
        )
        
        for container in ax.containers:
            ax.bar_label(container, fmt='%.1f')
            
        ax = axs[1] if sz == 1 else axs[1, i]

        x = frame["SPVAR num anneals"]
        y = frame["% fixed vars"]

        ax.bar(x, y, width=width)
        ax.set_ylabel("Percent of fixed vars")
        ax.set_xlabel("SPVAR number anneals")
        ax.hlines(
            y = 0,
            xmin=start_range,
            xmax=last_range
        )
        
        for container in ax.containers:
            ax.bar_label(container, fmt='%.1f')        
    
    plt.savefig(f"{dir_results}\\results.pdf")
    plt.close()

# Does some honest tests with params [total_num_anneals, SPVAR_num_anneals, fixing_threshhold, elite_threshold]
# and saves results to file named {total_num_anneals}_{fixing_threshold}_{elite_threshold}.csv in directory dir_results
# (SPVAR_num_anneals is an element of SPVAR_num_anneals_range);
# Also, if ignore_calced = False and csv file with results contains all nesesary information, it does not calc again
# Is draw_bars = True, function draw bar chart by and saves in to the same directory.
def test_different_num_anneals(
        total_num_anneals : int,
        SPVAR_num_anneals_range : range,
        fixing_threshold : float,
        elite_threshold : float,
        dir_results : str,
        data_file_path : str,
        ignore_calced : bool = False,
        draw_bars : bool = True):
    
    [h, J, _] = read_matrices.read_qubo_from_file(data_file_path)
    num_vars = len(h.keys())
    
    params = [
        spvar.SPVAR_test_honest_params
        (
            h,
            J,
            total_num_anneals,
            SPVAR_num_anneals,
            fixing_threshold,
            elite_threshold
        )
        for SPVAR_num_anneals in SPVAR_num_anneals_range
    ]

    result_path = f"{dir_results}\\{total_num_anneals}_{round(fixing_threshold, 1)}_{round(elite_threshold, 1)}.csv"

    if not ignore_calced and os.path.isfile(result_path):
        df = pd.read_csv(result_path)
        spvar_num_anneals_set = set(df["SPVAR num anneals"].to_list())
        fl = True
        for param in params:
            if (param.SPVAR_num_anneals not in spvar_num_anneals_set):
                fl = False
                break
        if fl:
            if (draw_bars):
                draw_plot(dir_results)
            return

    columns = ["SPVAR num anneals", "Result without SPVAR", "Result with SPVAR", "% fixed vars"]

    df = pd.DataFrame(columns=columns)

    for i in range(len(params)):
        param = params[i]

        s = spvar.SPVAR()
        [without_SPVAR, with_SPVAR, cnt_fixed] = s.test_honest(param)
        df.loc[i] = [param.SPVAR_num_anneals, without_SPVAR, with_SPVAR, round(cnt_fixed / num_vars * 100, 1)]
        
        fout = open(result_path, "w")
        df.to_csv(fout)
        fout.close()
    
    if draw_bars:
        draw_plot(dir_results)

#Runs test_different_num_anneals with all params from params_list
def test_multiple_params(params_list: list[tuple[int, range, float, float]],
                         dir_results : str,
                         data_file_path : str,
                         ignore_calced : bool = False):
    for (total_num_anneals, SPVAR_num_anneals_range, fixing_threshold, elite_threshold) in params_list:
        test_different_num_anneals(
            total_num_anneals,
            SPVAR_num_anneals_range,
            fixing_threshold,
            elite_threshold,
            dir_results,
            data_file_path,
            ignore_calced,
            False
        )
        draw_plot(dir_results)

# Run test_multiple_params for all files from data_dir
# Data_dir a relative path from directory test_data
# Directory with results for file f.csv is {DIR_RESULTS}\\{data_dir}\\{f} 
def test_multiple_params_over_directory(
        params: list[tuple[int, range, float, float]],
        data_dir : str):
    data_path = DIR_DATA + "\\" + data_dir
    results_path = DIR_RESULTS + "\\" + data_dir
    for data_file in os.listdir(data_path):
        data_file_path = data_path + "\\" + data_file
        file_name, file_extention = os.path.splitext(data_file)
        if (file_extention != ".csv"):
            continue
        results_file_dir = results_path + "\\" + file_name
        try:
            os.mkdir(results_file_dir)
        except OSError:
            pass
        
        test_multiple_params(params, results_file_dir, data_file_path)
def main():

    # sample of using:

    params = [
        tuple([200, range(100, 200, 10), 0.0, 0.2]),
        tuple([200, range(100, 200, 10), 0.1, 0.2]),
    ]

    test_multiple_params_over_directory(params, "qubo_matrices")

main()