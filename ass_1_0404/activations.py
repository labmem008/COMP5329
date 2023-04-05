import numpy as np


class Relu:
    def forward(self, x):
        self.x = x
        return np.maximum(0, x)
    
    def backward(self, eta):
        """
        x>0 grad = x
        x<=0 grad = 0
        """
        eta[self.x<=0] = 0
        return eta

class Gelu:
    def forward(self, x):
        self.out = 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x ** 3)))
        return self.out

    def backward(self, eta):
        self.grad = ((np.tanh((np.sqrt(2) * (0.044715 * self.out ** 3 + self.out)) / np.sqrt(np.pi)) + ((np.sqrt(2) * self.out * (
                    0.134145 * self.out ** 2 + 1) * ((1 / np.cosh(
            (np.sqrt(2) * (0.044715 * self.out ** 3 + self.out)) / np.sqrt(np.pi))) ** 2)) / np.sqrt(np.pi) + 1))) / 2
        return eta * self.grad

class Softmax:
    def forward(self, x):
        v = np.exp(x - x.max(axis=-1, keepdims=True))    
        return v / v.sum(axis=-1, keepdims=True)

    def backward(self, eta):
        # compute the gard with loss function would be way more easy (pred - labbel). 
        pass