import io_maxcut
import dimod
from functools import *
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

# reads QUBo from mile and returns it as an Ising problem
# return format: [h, J, offset]
def read_qubo_from_file(path : str) -> tuple[dict, dict, int]:
    f = open(path, "r")
    i = 0
    Q = dict()
    for line in f.readlines():
        coefs = list(map(float, line.split(',')))
        for j in range(len(coefs)):
            Q[(i, j)] = coefs[j]
        i += 1
    return dimod.qubo_to_ising(Q)

#Returns list of tuples [path, h, J]
def read_qubo_matrices(dir_name : str) -> list[tuple[str, dict, dict]]:
    testPathes = io_maxcut.files_in_directory(dir_name)

    result = []
    for testPath in testPathes:
        path = f"{dir_name}/" + testPath
        [h, J, _] = read_qubo_from_file(path)
        result.append(tuple([testPath, h, J]))
    
    return result

