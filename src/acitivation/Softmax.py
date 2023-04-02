import numpy as np


class Softmax:
    def __init__(self):
        pass
    
    def forward(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    # def backward(self, eta):
    #     self.grad = np.diag(self.out)-np.outer(self.out,self.out)
    #     return eta * self.grad