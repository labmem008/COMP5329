import numpy as np
from parameter import Parameter


class Linear:
    def __init__(self, in_dim, out_dim):
        # initialize W & b
        W = np.random.uniform(low=-np.sqrt(6.0 / (in_dim + out_dim)),high=np.sqrt(6.0 / (in_dim + out_dim)),size=(in_dim, out_dim))
        self.W = Parameter(W)
        self.b = Parameter(np.zeros(out_dim))

    def forward(self, x):
        # y = W * x + b
        self.x = x
        out = np.dot(x, self.W.data) + self.b.data
        return out
    
    def backward(self, eta):
        batch_size = eta.shape[0]
        self.W.grad = np.dot(self.x.T, eta) / batch_size
        self.b.grad = np.sum(eta, axis=0) / batch_size
        return np.dot(eta, self.W.data.T)