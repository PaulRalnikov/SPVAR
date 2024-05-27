import test_spvar
import pandas as pd
from functools import *
import os

DIR_OLD_RESULTS = "old_test_results"
DIR_RESULTS = "test_results"
DIR_DATA = "test_data"

def draw_plots_for_all_tasks(tasks_dir : str):
    path = DIR_RESULTS + "\\" + tasks_dir
    for dir in os.listdir(path):
        test_spvar.draw_plot(path + "\\" + dir)


# reworks all old results from directory dir to new format 
# e.g fills files in {DIR_RESULTS}\\mbo  by data in {DIR_OLD_RESULTS}\\mbo,
# files in {DIR_RESULTS}\\knapsack  by data in {DIR_OLD_RESULTS}\\knapcack and so on
def rework_result_directory(dir : str, need_build_csv : bool = True):
    
    old_dir_path = f"{DIR_OLD_RESULTS}\\{dir}"
    dirs = sorted(os.listdir(old_dir_path))
    
    if (len(dirs) == 0):
        print("No files for rework")
        return

    # results[file][params] - list with elements in format
    # [SPVAR_num_anneals, Result without SPVAR, Result with SPVAR, % fixed vars]
    # file - filename of QUBO promlem (contains extention)
    # params - string in format {total_num_anneals}_{fixing_threshhold}_{elite_threshold}
    if (need_build_csv) :
        results = dict()

        for d in dirs:

            d_path = old_dir_path + "\\" + d
            files = os.listdir(d_path)

            params = "_".join(d.split("_")[1:])
            for full_name in files:
                file_path = d_path + "\\" + full_name
                file_name, file_extention = os.path.splitext(full_name)
                if file_extention != ".csv":
                    continue
                
                SPVAR_num_anneals = int(file_name.split("_")[1])

                df = pd.read_csv(file_path, header=0, index_col=0)
                data_files = df.index.to_list()

                for data_file in data_files:
                    if data_file not in results:
                        results[data_file] = dict()
                    if params not in results[data_file]:
                        results[data_file][params] = []
                    without_SPVAR = df.loc[data_file]["Result without SPVAR"]
                    with_SPVAR = df.loc[data_file]["Result with SPVAR"]
                    p_fixed = df.loc[data_file]["% fixed vars"]

                    results[data_file][params].append([SPVAR_num_anneals, without_SPVAR, with_SPVAR, p_fixed])

        columns = ["SPVAR num anneals", "Result without SPVAR", "Result with SPVAR", "% fixed vars"]

        for data_file in results.keys():
            file_name, file_extention = os.path.splitext(data_file)
            dir_file_results = DIR_RESULTS + "\\" + dir + "\\" + file_name

            try:
                os.mkdir(dir_file_results)
            except OSError:
                pass

            for params in results[data_file].keys():
                df = pd.DataFrame(results[data_file][params], columns = columns)
                print(df)
                fout = open(f"{dir_file_results}\\{params}.csv", "w")
                df.to_csv(fout)
                fout.close()
    draw_plots_for_all_tasks(dir)
