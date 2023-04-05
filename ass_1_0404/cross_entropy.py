import numpy as np
from activations import Softmax


class CrossEntropy:
    def __init__(self):
        self.softmax = Softmax()

    def gradient(self):
        # gradient would be compute in __call__ first
        return self.grad

    def __call__(self, prediction, label):
        # label: One-hot vector, prediction: softmax vector
        prediction = self.softmax.forward(prediction)
        self.grad = prediction - label
        loss = - np.sum(label * np.log(prediction)) / label.shape[0]
        acc = np.argmax(prediction, axis=-1) == np.argmax(label, axis=-1)
        return acc, loss

        
        