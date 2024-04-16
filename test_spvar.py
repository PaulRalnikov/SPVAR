import spvar
import io_maxcut
import dimod
import pandas as pd

def read_ising_from_Gset_file(path : str) -> tuple[dict, dict, int]:
    g = io_maxcut.read_gset_graph(path)
    Q_mat = io_maxcut.graph_to_qubo(g)
    shape = Q_mat.shape

    Q_dict = dict()    
    for i in range(shape[0]):
        for j in range(shape[1]):
            Q_dict[(i, j)] = Q_mat[i][j]

    return dimod.qubo_to_ising(Q_dict)

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

# tests is a list of tuple[h, J, best_known]
def test_criteria(sample_size,
                  fixing_threshold,
                  elite_threshold,
                  anneal_duration_many,
                  anneal_duration_one,
                  tests,
                  output_path,
                  indexDataFrame,
                  is_min = True,
                  contains_best_known = True):
    print("start testing criterea:")
    print(f"sample_size: {sample_size}")
    print(f"fixing_threshold: {fixing_threshold}")
    print(f"elite_threshold: {elite_threshold}")
    print(f"anneal_duration_one: {anneal_duration_one}")
    print(f"anneal_duration_many: {anneal_duration_many}")

    results = []
    for test in tests:
        h = dict()
        J = dict()
        best_known = None
        if contains_best_known:
            [h, J, best_known] = test
        else:
            [h, J] = test
        
        sz = len(h)

        (before, after, cnt_fixed) = spvar.test_spvar(
            h,
            J,
            sample_size,
            fixing_threshold,
            elite_threshold,
            anneal_duration_one,
            anneal_duration_many
        )

        if not is_min:
            before = -before
            after = -after

        if contains_best_known:
            results.append([
                before,
                after,
                str(round(cnt_fixed / sz * 100), 1) + '%',
                best_known,
                str(round((before - best_known) / best_known * 100, 1)) + '%',
                str(round((after - best_known) / best_known * 100, 1)) + '%',  
                str(round((after - before) / before * 100, 1)) + '%',  
            ])
        else:
            results.append([
                before,
                after,
                str(round(cnt_fixed / sz * 100, 1)) + '%',
                str(round((after - before) / before * 100, 1)) + '%',  
            ])
    
    f = open(output_path, "a")
    
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
                "percent fixed varables",
                "best known",
                "(before - best) / best",
                "(after - best) / best",
                "(after - before) / before"
            ] if contains_best_known
            else [
                "result before SPVAR",
                "result after SPVAR",
                "percent fixed varables",
                "(after - before) / before"
            ],
            index = indexDataFrame
            ).to_string() + "\n",
    )
    f.write("=" * (161 if contains_best_known else 105) + "\n")
    f.close()


def test_criteria_Gset(sample_size,
                  fixing_threshold,
                  elite_threshold,
                  anneal_duration_many,
                  anneal_duration_one) :

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

    tests = []
    for testPath in testPathes:
        path = "Gset/" + testPath
        (h, J, _) = read_ising_from_Gset_file(path)
        tests.append([h, J, best_known_energy[testPath]])
    
    test_criteria(sample_size,
                  fixing_threshold,
                  elite_threshold,
                  anneal_duration_many,
                  anneal_duration_one,
                  tests,
                  "output.txt",
                  testPathes,
                  False)

def test_criteria_qubo_matrices(sample_size,
                  fixing_threshold,
                  elite_threshold,
                  anneal_duration_many,
                  anneal_duration_one):
    testPathes = io_maxcut.files_in_directory("qubo_matrices")

    tests = []
    for testPath in testPathes:
        path = "qubo_matrices/" + testPath
        Q = read_qubo_from_csv_file(path)
        [h, J, _] = dimod.qubo_to_ising(Q)
        tests.append([h, J])
    
    test_criteria(sample_size,
                  fixing_threshold,
                  elite_threshold,
                  anneal_duration_many,
                  anneal_duration_one,
                  tests,
                  "output_new.txt",
                  testPathes,
                  contains_best_known=False)


# test_criteria_Gset(20, 0.1, 0.2, 20, 20)

test_criteria_qubo_matrices(1000, 0.1, 0.2, 10000, 20)
test_criteria_qubo_matrices(1000, 0.0, 0.2, 10000, 20)
test_criteria_qubo_matrices(1000, 0.0, 0.1, 10000, 20)
test_criteria_qubo_matrices(1000, 0.1, 0.1, 10000, 20)

