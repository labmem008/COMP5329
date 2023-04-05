import numpy as np


class Relu:
    def __init__(self):
        pass

    def forward(self, x):
        self.x = x
        return np.maximum(self.x, 0)

    def backward(self, eta):
        eta[self.x <= 0] = 0
        return eta

