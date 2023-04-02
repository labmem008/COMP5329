import numpy as np
from loss import CrossEntropy
from acitivation.Relu import Relu
from acitivation.Softmax import Softmax
from layers.FullyConnection import FullyConnection

# adin6536@uni.sydney.edu.au

class MModel:
    def __init__(self):
        self.loss_fn = CrossEntropy()
        self.input_layer = FullyConnection(128, 256)
        self.hidden_layer1 = FullyConnection(256, 512)
        self.hidden_layer2 = FullyConnection(512, 128)
        self.output_layer = FullyConnection(128, 10)
        self.relu1 = Relu()
        self.relu2 = Relu()
        self.relu3 = Relu()

        self.softmax = Softmax()
        

    def forward(self, X, labels=None):
        out = self.relu1.forward(self.input_layer.forward(X))
        out = self.relu2.forward(self.hidden_layer1.forward(out))
        out = self.relu3.forward(self.hidden_layer2.forward(out))
        # out = self.output_layer.forward(out)
        self.out = self.softmax.forward(self.output_layer.forward(out))
        print(self.out)
        if labels is not None:
            labels = np.array(labels).reshape(-1)
            labels = np.eye(10)[labels]
            self.labels = labels
            self.loss= self.loss_fn(self.out, self.labels)
            print(self.loss)
            return self.out, self.labels
        return self.out

    def backward(self, lr):
        # prediction = np.max(self.out, axis=1, keepdims=True)
        eta = self.loss_fn.gradient(self.out, self.labels)
        eta = self.output_layer.backward(eta, lr)
        eta = self.relu3.backward(eta)
        eta = self.hidden_layer2.backward(eta, lr)
        eta = self.relu2.backward(eta)
        eta = self.hidden_layer1.backward(eta, lr)
        eta = self.relu1.backward(eta)
        eta = self.input_layer.backward(eta, lr)

