import classes as cl
import numpy as np
from methods import Korpelevich, KorpelevichAdapt, MalTam, MalTamAdapt, InitializeSpace
import time
import random
from copy import copy, deepcopy
import matplotlib.pyplot as plt
import matplotlib.patches as patches

mark_amnt = 6
empty_links_ratio = 1
total_shipment = 10
dim = 0

# link_const = [1, 10]
sup_coef = [0, 4]
dem_coef = [0, 4]
trans_coef = [0, 4]
sup_const = [0, 10]
dem_const = [10 ** mark_amnt, 10 ** mark_amnt + 50]
trans_const = [0, 5]

realistic_links = True
bipartial_links = False
links = np.zeros([mark_amnt, mark_amnt])
# link_cost = np.ones([mark_amnt, mark_amnt]) * -1
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

# def GenerateReqiredShipment():
#     remaining_dem_shipment = total_shipment
#     remaining_sup_shipment = total_shipment
#     for i in range(0, mark_amnt - 1):
#         req_dem[i] = random.randint(0, remaining_dem_shipment)
#         req_sup[i] = random.randint(0, remaining_sup_shipment)
#         remaining_dem_shipment -= req_dem[i]
#         remaining_sup_shipment -= req_sup[i]
#     req_dem[mark_amnt - 1] = remaining_dem_shipment
#     req_sup[mark_amnt - 1] = remaining_sup_shipment

#     print("\nRequired Supply Shipment:\n" + str(req_sup))
#     print("\nRequired Demand Shipment:\n" + str(req_dem))


def GenerateDependencies():
    tmp_sup_price = np.ndarray([mark_amnt, dim])
    tmp_dem_price = np.ndarray([mark_amnt, dim])

    global dem_price
    global sup_price
    global trans_price
    global dem_consts
    global sup_consts
    global trans_consts

    sup_price = np.ndarray([dim, dim])
    dem_price = np.ndarray([dim, dim])
    trans_price = np.ndarray([dim, dim])

    sup_consts = np.ndarray(dim)
    dem_consts = np.ndarray(dim)
    trans_consts = np.ndarray(dim)

    for i in range(0, mark_amnt):
        for j in range(0, dim):
            tmp_sup_price[i, j] = random.randint(sup_coef[0], sup_coef[1])
            tmp_dem_price[i, j] = random.randint(dem_coef[0], dem_coef[1])

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
    print("Supply Price Coefs: " + str(sup_price))
    print("Demand Price Coefs: " + str(dem_price))
    print("Transaction Price Coefs: " + str(trans_price))

    print("Supply Price Consts: " + str(sup_consts))
    print("Demand Price Consts: " + str(dem_consts))
    print("Transaction Price Consts: " + str(trans_consts))


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
    print("\nPaths:\n" + str(paths))

def proceedPath(root, mark_set):
        for j in mark_set:
            if links[root[1], j]:
                path = copy(root)
                # path = deepcopy(root)
                # path[3].append(j)
                path[1] = j
                # path[2] += link_cost[root[1], j]
                paths.append(path)

                new_set = mark_set.copy()
                new_set.remove(j)
                proceedPath(path, new_set)

def InitializeLinks():
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


    # for i in range(0, mark_amnt):
    #     for j in range(0, mark_amnt):
    #         if links[i, j]:
    #             link_cost[i, j] = random.randint(link_const[0], link_const[1])
    # print("\nLinks' cost:\n" + str(link_cost))

    print("\nLinks:\n" + str(links))

def operator1(point):
    return (sup_price + trans_price - dem_price).dot(point) \
           + (sup_consts + trans_consts - dem_consts)

# def operator2(point):
#     sup = np.zeros(mark_amnt)
#     dem = np.zeros(mark_amnt)
#     for i in range(0, dim):
#             sup[paths[i][0]] += point[i]
#             dem[paths[i][1]] += point[i]

#     res = np.ndarray(dim)
#     for i in range(0, mark_amnt):
#         res[i] = sup[i] - req_sup[i] + dem[i] - req_dem[i]
#     return res
#     return point

# def error(point):
#     print("Error: " + str((sup_price + trans_price - dem_price).dot(point) \
#            + (sup_consts + trans_consts - dem_consts)))

def DisplayResults():
    fig, ax = plt.subplots()
    r = 0.5
    bord = mark_amnt + 2 * r
    plt.xlim(-bord, bord)
    plt.ylim(-bord, bord)
    for i in range(0, mark_amnt):
        ax.add_artist(plt.Circle((mark_amnt * np.cos(2 * i * np.pi / mark_amnt), mark_amnt * np.sin(2 * i * np.pi / mark_amnt)), radius=r))
    plt.show()

def SpatialPriceEqulibrium():
    # GenerateReqiredShipment()
    InitializeLinks()
    GeneratePaths()
    GenerateDependencies()
    DisplayResults()

    A1 = cl.Operator(operator1, norm)
    # A2 = cl.Operator(operator2)
    space = InitializeSpace(dim)

    print("Korpelevich Method:")
    res = Korpelevich(space, A1)

    print("Korpelevich Adapt Method:")
    res = KorpelevichAdapt(space, A1)

    print("MalTam Method:")
    res = MalTam(space, A1)

    print("MalTam Adapt Method:")
    res = MalTamAdapt(space, A1)