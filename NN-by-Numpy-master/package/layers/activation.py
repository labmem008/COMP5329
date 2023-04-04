from .layer import Layer
import numpy as np


class Relu(Layer):
    def forward(self, x):
        self.x = x
        return np.maximum(0, x)

    def backward(self, eta):
        eta[self.x<=0] = 0
        return eta

class Gelu(Layer):
    def forward(self, x):
        self.out = 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x ** 3)))

    def backward(self, eta):
        self.grad = ((np.tanh((np.sqrt(2) * (0.044715 * self.out ** 3 + self.out)) / np.sqrt(np.pi)) + ((np.sqrt(2) * self.out * (
                    0.134145 * self.out ** 2 + 1) * ((1 / np.cosh(
            (np.sqrt(2) * (0.044715 * self.out ** 3 + self.out)) / np.sqrt(np.pi))) ** 2)) / np.sqrt(np.pi) + 1))) / 2
        return eta * self.grad


class Softmax(Layer):
    def forward(self, x):
        '''
        x.shape = (N, C)
        接收批量的输入，每个输入是一维向量
        计算公式为：
        a_{ij}=\frac{e^{x_{ij}}}{\sum_{j}^{C} e^{x_{ij}}}
        '''
        v = np.exp(x - x.max(axis=-1, keepdims=True))    
        return v / v.sum(axis=-1, keepdims=True)
    
    def backward(self, y):
        # 一般Softmax的反向传播和CrossEntropyLoss的放在一起
        pass