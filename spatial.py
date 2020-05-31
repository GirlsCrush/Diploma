import classes as cl
import numpy as np
from methods import Korpelevich, KorpelevichAdapt, MalTam, MalTamAdapt, InitializeSpace
import time
import random
from copy import copy, deepcopy
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys

mark_amnt = None
mark_amnt_const = [3, 7]
empty_links_ratio = 1
dim = None
input_file = None

# link_const = [1, 10]
sup_coef = [1, 4]
dem_coef = [1, 4]
trans_coef = [1, 2]
sup_const = [0, 10]
dem_const = None
trans_const = [0, 5]

realistic_links = True
bipartial_links = False
links = None
paths = []

# req_sup = np.zeros(mark_amnt)
# req_dem = np.zeros(mark_amnt)

dem_price = None
sup_price = None
trans_price = None
dem_consts = None
sup_consts = None
trans_consts = None
req_sup_n_dem = None
norm = None

def GenerateDependencies():
    tmp_sup_price = np.ndarray([mark_amnt, dim])
    tmp_dem_price = np.ndarray([mark_amnt, dim])

    global dem_price
    global sup_price
    global trans_price
    global dem_consts
    global sup_consts
    global trans_consts
    global input_file

    sup_price = np.ndarray([dim, dim])
    dem_price = np.ndarray([dim, dim])
    trans_price = np.ndarray([dim, dim])

    sup_consts = np.ndarray(dim)
    dem_consts = np.ndarray(dim)
    trans_consts = np.ndarray(dim)

    dem_const = [
        (dem_coef[1] - sup_coef[0] - trans_coef[0]) * dim,
        (dem_coef[1] - sup_coef[0] - trans_coef[0]) * (dim + 1)
    ]

    for i in range(0, mark_amnt):
        for j in range(0, dim):
            tmp_sup_price[i, j] = random.randint(sup_coef[0], sup_coef[1])
            tmp_dem_price[i, j] = -random.randint(dem_coef[0], dem_coef[1])

    for i in range(0, dim):
        for j in range(0, dim):
            trans_price[i, j] = random.randint(trans_coef[0], trans_coef[1])
        sup_price[i, :] = tmp_sup_price[paths[i][0], :]
        dem_price[i, :] = tmp_dem_price[paths[i][1], :]

        sup_consts[i] = random.randint(sup_const[0], sup_const[1])
        dem_consts[i] = random.randint(dem_const[0], dem_const[1])
        trans_consts[i] = random.randint(trans_const[0], trans_const[1])

    global norm
    norm = np.linalg.norm(sup_price + trans_price - dem_price)

    input_file.write("\nSupply Price Coefs:\n" + str(sup_price))
    input_file.write("\nDemand Price Coefs:\n" + str(dem_price))
    input_file.write("\nTransaction Price Coefs:\n" + str(trans_price))

    input_file.write("\nSupply Price Consts:\n" + str(sup_consts))
    input_file.write("\nDemand Price Consts:\n" + str(dem_consts))
    input_file.write("\nTransaction Price Consts:\n" + str(trans_consts))
    input_file.write("\nMarket amount:\n" + str(mark_amnt))
    input_file.write("\nDim:\n" + str(dim))


def GeneratePaths():
    mark_set = set(range(0, mark_amnt))
    for i in mark_set:
        path = [i, i
                # , 0
                # , [i]
        ]
        new_set = mark_set.copy()
        new_set.remove(i)
        proceedPath(path, new_set)
    global dim
    dim = len(paths)
    input_file.write("\n\nPaths:\n" + str(paths))

def proceedPath(root, mark_set):
        for j in mark_set:
            if links[root[1], j]:
                path = copy(root)
                path[1] = j
                paths.append(path)

                new_set = mark_set.copy()
                new_set.remove(j)
                proceedPath(path, new_set)

def InitializeLinks():
    global links
    links = np.zeros([mark_amnt, mark_amnt])
    if bipartial_links:
        for i in range(mark_amnt // 2, mark_amnt):
            for j in range(0, mark_amnt // 2):
                links[i, j] = 1
        # links[mark_amnt // 2 - 1, mark_amnt // 2 - 1] = 0
    else:
        for i in range(0, mark_amnt):
            for j in range(0, mark_amnt):
                if i != j and random.uniform(-empty_links_ratio, 1) > 0:
                    links[i, j] = 1

        if realistic_links:
            for i in range(0, mark_amnt):
                if not sum(links[i, :]):
                    j = random.randrange(0, mark_amnt)
                    while j == i :
                        j = random.randrange(0, mark_amnt)
                    links[i, j] = 1
                if not sum(links[:, i]):
                    j = random.randrange(0, mark_amnt)
                    while j == i :
                        j = random.randrange(0, mark_amnt)
                    links[j, i] = 1
    DisplayLinks()
    input_file.write("\n\nLinks:\n" + str(links))

def operator1(point):
    return (sup_price + trans_price - dem_price).dot(point) \
           + (sup_consts + trans_consts - dem_consts)

def DisplayLinks():
    fig, ax = plt.subplots()
    kw = dict(arrowstyle="Simple,tail_width=0.5,head_width=5,head_length=8", color="k")
    r = 0.5 + mark_amnt / 10
    dist = 2 * mark_amnt
    bord = dist + 2 * r
    plt.axis('off')
    plt.xlim(-bord, bord)
    plt.ylim(-bord, bord)

    centers = []
    for i in range(0, mark_amnt):
        centers.append(np.array([dist * np.cos(2 * i * np.pi / mark_amnt), dist * np.sin(2 * i * np.pi / mark_amnt)]))

    letter_vector = np.array([0.2, 0.3])

    for i in range(0, mark_amnt):
        for j in range(0, mark_amnt):
            if links[i, j]:
                ax.add_patch(patches.FancyArrowPatch(
                    (centers[i] + (centers[j] - centers[i]) / np.linalg.norm(centers[j] - centers[i]) * r).tolist(),
                    (centers[j] + (centers[i] - centers[j]) / np.linalg.norm(centers[i] - centers[j]) * r).tolist(),\
                    connectionstyle="arc3,rad=" + str(1. / mark_amnt), **kw))
        ax.text((centers[i] - letter_vector)[0], (centers[i] - letter_vector)[1], str(i), fontweight='bold')
        ax.add_artist(plt.Circle(centers[i], radius=r, fill=False))

    fig.savefig("links" + str(mark_amnt) + ".png", bbox_inches='tight')

def SpatialPriceEqulibrium():
    global mark_amnt
    global dim
    global input_file
    global dem_const
    global links

    l_list = [0.5, 0.25, 0.125, 0.0625]
    tau_list = [0.95, 0.9, 0.7, 0.5]
    # l_list = [0.5]
    # tau_list = [0.9]
    methods = [
        ["Korp", Korpelevich, False],
        ["KorpAdapt", KorpelevichAdapt, True],
        ["MalTam", MalTam, False],
        ["MalTamAdapt", MalTamAdapt, True]
    ]

    for mark_amnt in range(mark_amnt_const[0], mark_amnt_const[1]):
        input_file = open("Input_Mark_" + str(mark_amnt) + ".txt", "w")

        InitializeLinks()
        GeneratePaths()
        GenerateDependencies()
        # DisplayResults()


        input_file.close()
        print("\nMarket amount: " + str(mark_amnt))
        print("\nDim: " + str(dim))

        A1 = cl.Operator(operator1, norm)
        space = InitializeSpace(dim)
        for method in methods:
            min_iter_amnt = sys.maxsize
            for l in l_list:
                for tau in tau_list:
                    res, iter_amnt, el_time = method[1](space, A1, l, tau)
                    if min_iter_amnt > iter_amnt:
                        min_iter_amnt = iter_amnt
                        f_name = method[0] + "_Mark_" + str(mark_amnt)
                        f = open(f_name + ".txt", "w")
                        f.write("\nResult:\n" + str(res))
                        f.write("\nIteration amount:\n" + str(iter_amnt))
                        f.write("\nElapsed time:\n" + str(el_time))
                        f.write("\nLambda:\n" + str(l))
                        f.write("\nTau:\n" + str(tau))
                        f.close()
                    if not method[2]:
                        break
