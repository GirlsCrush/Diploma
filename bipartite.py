import classes as cl
import numpy as np
from methods import Korpelevich, KorpelevichAdapt, MalTam, MalTamAdapt
import time
import sys

sup_amnt = 2
dem_amnt = 2
dim = sup_amnt * dem_amnt
# dim = (sup_amnt + 1) * (dem_amnt + 1) - 1


sup_consts = np.array([2, 2, 3, 3])
dem_consts = np.array([28.75, 41, 28.75, 41])
trans_consts = np.array([1, 1.5, 15, 10])
dem_price = np.array([
    [-2, -1, -2, -1],
    [-1, -4, -1, -4],
    [-2, -1, -2, -1],
    [-1, -4, -1, -4]
])
sup_price = np.array([
    [5, 5, 1, 1],
    [5, 5, 1, 1],
    [1, 1, 2, 2],
    [1, 1, 2, 2],
])
trans_price = np.array([
    [1, 0.5, 0, 0],
    [0,   2, 0, 1],
    [2,   0, 3, 0],
    [0,   1, 0, 2]
])

req_sup_n_dem = [3, 2, 1.5, 3.5]
# req_sup_n_dem = [100, 0, 0, 100]

norm = np.linalg.norm(sup_price + trans_price - dem_price)

def operator1(point):
    return (sup_price + trans_price - dem_price).dot(point) \
           + (sup_consts + trans_consts - dem_consts)

def operator2(point):
    sup = np.zeros(sup_amnt)
    dem = np.zeros(dem_amnt)
    for i in range(0, sup_amnt):
        for j in range(0, dem_amnt):
            sup[i] += point[i * sup_amnt + j]
            dem[j] += point[i * sup_amnt + j]

    res = np.ndarray(dim)
    for i in range(0, sup_amnt):
        for j in range(0, dem_amnt):
            res[i * sup_amnt + j] = sup[i] - req_sup_n_dem[i] + dem[j] - req_sup_n_dem[sup_amnt + j]
    return res

# def error(point):
#     sup_n_dem = np.zeros(sup_amnt + dem_amnt)
#     for i in range(0, sup_amnt):
#         for j in range(0, dem_amnt):
#             sup_n_dem[i]            += point[i * sup_amnt + j]
#             sup_n_dem[sup_amnt + j] += point[i * sup_amnt + j]
#     print("Error: " + str(np.linalg.norm(req_sup_n_dem - sup_n_dem)))

def error(point):
    return np.linalg.norm(point - np.array([1.5, 1.5, 0, 2]))

def BipatitePriceEqulibrium():
    print("Norm: " + str(norm))
    A1 = cl.Operator(operator1, norm)
    # A2 = cl.Operator(operator2)
    spaces = list()

    for i in range(0, dim):
        spaces.append(cl.HalfSpace(dim, 0, i, False))
    space = cl.HalfSpaceSet(spaces)
    space.dim = dim

    l_list = np.arange(0.5, 0, -0.01)
    tau_list = np.arange(0.5, 1, 0.01)
    # l_list = [0.5]
    # tau_list = [0.9]
    methods = [
        ["Korp", Korpelevich, False],
        ["KorpAdapt", KorpelevichAdapt, True],
        ["MalTam", MalTam, False],
        ["MalTamAdapt", MalTamAdapt, True]
    ]

    for method in methods:
        min_iter_amnt = sys.maxsize
        for l in l_list:
            for tau in tau_list:
                res, iter_amnt, el_time = method[1](space, A1, l, tau)
                if min_iter_amnt > iter_amnt:
                    min_iter_amnt = iter_amnt
                    f_name = "Bipartitie_" + method[0]
                    f = open(f_name + ".txt", "w")
                    f.write("\nResult:\n" + str(res))
                    f.write("\nIteration amount:\n" + str(iter_amnt))
                    f.write("\nElapsed time:\n" + str(el_time))
                    f.write("\nError:\n" + str(error(res)))
                    f.write("\nLambda:\n" + str(l))
                    f.write("\nTau:\n" + str(tau))
                    f.close()
                if not method[2]:
                    break


    # print("Ordinary Method:")
    # normal_res = Korpelevich2(space, A1, A2)
    # error(normal_res[0])

    # print("Modified Method:")
    # modif_res = Korpelevich2(space, A1, A2, True)
    # error(modif_res[0])
