import numpy as np

class Space:
    def __init_(self, dim : int):
        self.dim = dim

class HalfSpace(Space):
    def __init__(self, dim: int, val: float, valDim: int, isLess: bool) :
        # super().__init__(self, dim)
        self.val = val
        self.valDim = valDim
        self.isLess = isLess

    def proj(self, point: np.array):
        res = np.copy(point)
        if self.isLess and res[self.valDim] > self.val or \
           not self.isLess and res[self.valDim] < self.val:
            res[self.valDim] = self.val
        return res

class Sphere(Space):
    def __init__(self, center: np.array, radius: float, isInner = True) :
        # super().__init__(center.shape[0])
        self.center = center
        self.radius = radius
        self.isInner = isInner

    def proj(self, point: np.array):
        res = np.copy(point)
        if self.isInner and np.linalg.norm(self.center - res) > self.radius or \
           not self.isInner and np.linalg.norm(self.center - res) < self.radius:
            res = self.center + (res - self.center) * self.radius / np.linalg.norm(self.center - res)
        return res


class HalfSpaceSet(Space):
    def __init__(self, spaces: list) :
        # super().__init__(spaces[0].dim)
        self.spaces = spaces

    def proj(self, point: np.array):
        res = np.copy(point)
        for space in self.spaces:
            res = space.proj(res)
        return res

class SpaceSet(Space):
    def __init__(self, spaces: list, precision = 10e-5) :
        # super().__init__(spaces[0].dim)
        self.spaces = spaces
        self.precision = precision

    def proj(self, point: np.array):
        prev = np.copy(point)
        prev[0] = prev[0] + self.precision
        res = np.copy(point)

        while np.linalg.norm(prev - res) >= self.precision:
            prev = np.copy(res)
            res = np.zeros(res.shape)
            for space in self.spaces:
                res = res + space.proj(prev)
            res = res / len(self.spaces)

        return res

class Operator:
    def __init__(self, op = None, op_norm = 1):
        self.op = op
        self.op_norm = op_norm
    def dot(self, point):
        if self.op is None:
            return point
        if type(self.op) is np.array:
            return self.op.dot(point)
        return self.op(point)
    def norm(self):
        if type(self.op) is np.array:
            return np.linalg.norm(self.op)
        return self.op_norm