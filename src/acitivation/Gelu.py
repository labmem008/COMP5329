import numpy as np


class Gelu:
    def __init__(self):
        pass

    def forward(self, x):
        self.out = 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x ** 3)))


    def backward(self, eta):
        self.grad = ((np.tanh((np.sqrt(2) * (0.044715 * self.out ** 3 + self.out)) / np.sqrt(np.pi)) + ((np.sqrt(2) * self.out * (
                    0.134145 * self.out ** 2 + 1) * ((1 / np.cosh(
            (np.sqrt(2) * (0.044715 * self.out ** 3 + self.out)) / np.sqrt(np.pi))) ** 2)) / np.sqrt(np.pi) + 1))) / 2
        return eta * self.grad