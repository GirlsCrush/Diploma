import classes as cl
import numpy as np
import time

precision = 1e-5

def InitializeSpace(dim):
    spaces = []
    for i in range(0, dim):
        spaces.append(cl.HalfSpace(dim, 0, i, False))
        # spaces.append(cl.HalfSpace(dim, 10e+10, i, True))
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

def Korpelevich(space, A, l_coef = 0.5, tau = 0):
    res = np.zeros(space.dim)
    l = l_coef / A.norm()
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

def KorpelevichAdapt(space, A, l = 0.5, tau = 0.9):
    res = np.zeros(space.dim)
    prev = np.copy(res)
    prev[0] = prev[0] + 2 * precision
    iter_amnt = 0
    A_dots = []

    start = time.time()
    while True:
        iter_amnt += 1
        prev = np.copy(res)
        A_dots.clear()
        A_dots.append(A.dot(prev))
        y_n = space.proj(prev - A_dots[0] * l)
        if np.linalg.norm(y_n - res) < precision:
            break
        A_dots.append(A.dot(y_n))
        res = space.proj(prev - A_dots[1] * l)

        tmp = (A_dots[0] - A_dots[1]).dot(res - y_n)
        if tmp != 0:
                l = min(l, tau / 2 * \
                    ((np.linalg.norm(prev - y_n)) ** 2 + (np.linalg.norm(res - y_n)) ** 2) / tmp)
        # if not iter_amnt % 10000:
        #     print (iter_amnt)
        #     print (res)

    end = time.time()

    print("Point: " + str(res) \
          + ", iterations amount: " + str(iter_amnt) \
          + ", elapsed time: " + str(end - start))
    return res, iter_amnt, end - start

def MalTam(space, A, l_coef = 0.5, tau = 0):
    res = np.zeros(space.dim)
    prev = np.ones(space.dim)
    iter_amnt = 0
    l = l_coef / A.norm()
    A_dots = [0, A.dot(prev)]

    start = time.time()
    while True:
        iter_amnt += 1
        A_dots.pop(0)
        A_dots.append(A.dot(res))
        nxt = space.proj(res + l * (-2 * A_dots[1] + A_dots[0]))
        if np.linalg.norm(res - prev) < precision and np.linalg.norm(res - nxt) < precision:
            break
        prev = res
        res = nxt
        # if not iter_amnt % 10000:
        #     print (iter_amnt)
        #     print (res)

    end = time.time()

    print("Point: " + str(res) \
          + ", iterations amount: " + str(iter_amnt) \
          + ", elapsed time: " + str(end - start))
    return res, iter_amnt, end - start

def MalTamAdapt(space, A,  l = 1e-1, tau = 0.1):
    res = np.zeros(space.dim)
    prev = np.ones(space.dim)
    l_prev = l / 10
    iter_amnt = 0
    A_dots = [A.dot(prev), A.dot(res)]

    start = time.time()
    while True:
        iter_amnt += 1
        nxt = space.proj(res - l * A_dots[1] - l_prev * (A_dots[0] - A_dots[1]))
        if np.linalg.norm(res - prev) < precision and np.linalg.norm(res - nxt) < precision:
            break
        A_dots.pop(0)
        A_dots.append(A.dot(nxt))
        l_prev = l
        l = min(l, tau * np.linalg.norm(nxt - res) / np.linalg.norm(A_dots[1] - A_dots[0]))
        prev = res
        res = nxt
        # if not iter_amnt % 10000:
        #     print (iter_amnt)
        #     print (res)

    end = time.time()

    print("Point: " + str(res) \
          + ", iterations amount: " + str(iter_amnt) \
          + ", elapsed time: " + str(end - start))
    return res, iter_amnt, end - start
