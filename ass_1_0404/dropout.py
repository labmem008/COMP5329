import numpy as np

class Dropout:
    def __init__(self, drop_rate, is_test=False, **kwargs):
        self.drop_rate = drop_rate
        self.fix_value = 1 / (1 - drop_rate)
        self.is_test = is_test

    def forward(self, x):
        if self.is_test:
            return x
        else:
            self.mask = np.random.uniform(0, 1, x.shape) > self.drop_rate
            return x * self.mask * self.fix_value

    def backward(self, eta):
        if self.is_test:
            return eta
        else:
            return eta * self.mask