import numpy as np
import os


def files_in_directory(folder: str):
    recs = os.listdir(folder)
    lst = []
    for rec in recs:
        if os.path.isfile(os.path.join(folder, rec)):
            lst.append(rec)
    lst.sort()
    return lst


def read_gset_graph(path: str):
    with open(file=path) as my_file:
        n, nz = [int(st) for st in my_file.readline().split()]
        g = np.zeros((n, n))
        for k in range(nz):
            i, j, c = my_file.readline().split()
            i = int(i)-1
            j = int(j)-1
            c = int(c)
            g[i][j] = c
            g[j][i] = c
    return g


def graph_to_qubo(g: np.ndarray):
    n = len(g)
    q = g.copy()
    for i in range(n):
        for k in range(n):
            q[i][i] -= g[i][k]
    return q
