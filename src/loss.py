import numpy as np


class CrossEntropy:
    def __call__(self, prediction, label):
        # print(prediction.shape)
        prediction = np.array(prediction)
        # print(prediction[:10])
        self.num_data, num_class = prediction.shape
        # print(self.num_data)
        prediction += np.array([1e-15])
        log_p = np.log(prediction)
        # label_onehot = np.eye(num_class)[label]
        # print(label.shape, log_p.shape)
        # print(- np.sum(label * log_p))
        loss = - np.sum(label * log_p) / self.num_data
        
        # loss = np.multiply(np.log(prediction), label) + np.multiply((1 - label), np.log(1 - prediction)) #cross entropy
        # cost = -np.sum(loss)/num_data #num of examples in batch is m
        return loss
    
    def gradient(self, prediction, label):
        self.grad = prediction - label
        return self.grad/self.num_data
