import numpy as np


class SGD(object):
    def __init__(self, parameters, lr, decay=0):
        self.parameters = parameters
        self.lr = lr
        self.decay_rate = 1.0 - decay

    def update(self):
        for p in self.parameters:
            p.data *= self.decay_rate
            p.data -= self.lr * p.grad


class Momentum(object):
    def __init__(self, parameters, lr, decay=0, beta=0.9):
        self.lr = lr
        self.decay_rate = 1.0 - decay
        self.beta = beta
        self.parameters = parameters
        self.accmulated_grads = [np.zeros(p.data.shape) for p in self.parameters]

    def update(self):
        for p, grad in zip(self.parameters, self.accmulated_grads):
            p.data *= self.decay_rate
            grad = self.beta * grad + (1 - self.beta) * p.grad
            p.data -= self.lr * grad