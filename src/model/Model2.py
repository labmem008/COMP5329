import numpy as np
from loss import CrossEntropy
from acitivation.Relu import Relu
from acitivation.Softmax import Softmax
from layers.FullyConnection import FullyConnection

# adin6536@uni.sydney.edu.au

class MModel2:
    def __init__(self):
        self.loss_fn = CrossEntropy()
        self.input_layer = FullyConnection(128, 64)
        self.output_layer = FullyConnection(64, 10)
        self.relu = Relu()
        self.softmax = Softmax()
        

    def forward(self, X, labels=None):
        out = self.relu.forward(self.input_layer.forward(X))
        self.out = self.softmax.forward(self.output_layer.forward(out))
        # print(self.out)
        if labels is not None:
            labels = np.array(labels).reshape(-1)
            labels = np.eye(10)[labels]
            self.labels = labels
            self.loss= self.loss_fn(self.out, self.labels)
            # print(self.loss)
            return self.out, self.labels
        return self.out

    def backward(self, lr):
        # prediction = np.max(self.out, axis=1, keepdims=True)
        eta = self.loss_fn.gradient(self.out, self.labels)
        eta = self.output_layer.backward(eta, lr)
        eta = self.relu.backward(eta)
        eta = self.input_layer.backward(eta, lr)

