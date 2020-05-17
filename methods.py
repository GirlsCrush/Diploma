import classes as cl
import numpy as np
import time

precision = 1e-6
l_coef = 0.5

def InitializeSpace(dim):
    spaces = []
    for i in range(0, dim):
        spaces.append(cl.HalfSpace(dim, 0, i, False))
        spaces.append(cl.HalfSpace(dim, 10e+10, i, True))
    space = cl.HalfSpaceSet(spaces)
    space.dim = dim
    return space

# def Korpelevich2(space, A1, A2, modified = False):
#     res = np.zeros(space.dim)
#     prev = np.copy(res)
#     prev[0] = prev[0] + 2 * precision
#     alpha = 1.

#     if modified:
#         l = 1
#         tau = 0.9
#     else:
#         l = l_coef / A1.norm()
#         print ("l = " + str(l))

#     start = time.time()
#     while np.linalg.norm(res - prev) >= precision:
#         prev = res

#         y_n = space.proj(prev - A1.dot(prev) * l)
#         z_n = space.proj(prev - A1.dot(y_n) * l)
#         res = z_n - A2.dot(z_n) * alpha

#         alpha = 1. / (1. / alpha + 1)
#         # alpha = 1. / (1. / (alpha * alpha) + 1) ** 0.5
#         tmp = np.linalg.norm(A1.dot(prev) - A1.dot(y_n))
#         if modified and tmp != 0:
#                 l = min(l, tau * np.linalg.norm(prev - y_n) / tmp)
#         # if not round(1. / alpha) % 10:
#         #     print (round(1. / alpha))
#         #     print (res)
#     end = time.time()

#     print("Point: " + str(res) \
#           + ", iterations amount: " + str(round(1. / alpha)) \
#           + ", elapsed time: " + str(end - start))
#     return res, round(1. / alpha), end - start

def Korpelevich(space, A):
    res = np.zeros(space.dim)
    l = l_coef / A.norm()
    print ("l = " + str(l))
    iter_amnt = 0
    start = time.time()
    while True:
        iter_amnt += 1
        y_n = space.proj(res - A.dot(res) * l)
        if np.linalg.norm(y_n - res) < precision:
            break
        res = space.proj(res - A.dot(y_n) * l)
        if not iter_amnt % 10000:
            print (iter_amnt)
            print (res)
    end = time.time()

    print("Point: " + str(res) \
          + ", iterations amount: " + str(iter_amnt) \
          + ", elapsed time: " + str(end - start))
    return res, iter_amnt, end - start

def KorpelevichAdapt(space, A, isFirst = False):
    res = np.zeros(space.dim)
    prev = np.copy(res)
    prev[0] = prev[0] + 2 * precision
    l = 1
    tau = 0.9
    iter_amnt = 0

    start = time.time()
    while True:
        iter_amnt += 1
        prev = np.copy(res)
        y_n = space.proj(res - A.dot(res) * l)
        if np.linalg.norm(y_n - res) < precision:
            break
        res = space.proj(res - A.dot(y_n) * l)

        if not isFirst:
            tmp = (A.dot(prev) - A.dot(y_n)).dot(res - y_n)
            if tmp != 0:
                    l = min(l, tau / 2 * \
                        ((np.linalg.norm(prev - y_n)) ** 2 + (np.linalg.norm(res - y_n)) ** 2) / tmp)
        else:
            tmp = np.linalg.norm(A.dot(prev) - A.dot(y_n))
            if tmp != 0:
                l = min(l, tau * (np.linalg.norm(prev - y_n) / tmp))
        if not iter_amnt % 10000:
            print (iter_amnt)
            print (res)

    end = time.time()

    print("Point: " + str(res) \
          + ", iterations amount: " + str(iter_amnt) \
          + ", elapsed time: " + str(end - start))
    return res, iter_amnt, end - start

def MalTam(space, A):
    res = np.zeros(space.dim)
    prev = np.ones(space.dim)
    l = 0.25 / A.norm()
    iter_amnt = 0

    start = time.time()
    while True:
        iter_amnt += 1
        nxt = space.proj(res + l * (-2 * A.dot(res) + A.dot(prev)))
        if np.linalg.norm(res - prev) < precision and np.linalg.norm(res - nxt) < precision:
            break
        prev = res
        res = nxt
        if not iter_amnt % 10000:
            print (iter_amnt)
            print (res)

    end = time.time()

    print("Point: " + str(res) \
          + ", iterations amount: " + str(iter_amnt) \
          + ", elapsed time: " + str(end - start))
    return res, iter_amnt, end - start

def MalTamAdapt(space, A):
    res = np.zeros(space.dim)
    prev = np.ones(space.dim)
    tau = 0.1
    l = 1e-1
    l_prev = l / 10
    iter_amnt = 0

    start = time.time()
    while True:
        iter_amnt += 1
        nxt = space.proj(res - l * A.dot(res) - l_prev * (A.dot(res) - A.dot(prev)))
        if np.linalg.norm(res - prev) < precision and np.linalg.norm(res - nxt) < precision:
            break
        l_prev = l
        l = min(l, tau * np.linalg.norm(nxt - res) / np.linalg.norm(A.dot(nxt) - A.dot(res)))
        prev = res
        res = nxt
        if not iter_amnt % 10000:
            print (iter_amnt)
            print (res)

    end = time.time()

    print("Point: " + str(res) \
          + ", iterations amount: " + str(iter_amnt) \
          + ", elapsed time: " + str(end - start))
    return res, iter_amnt, end - start
