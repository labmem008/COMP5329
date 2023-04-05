import numpy as np


class FullyConnection:
    def __init__(self, input_dim, output_dim) -> None:
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = np.random.rand(input_dim, output_dim, )
        self.bias = np.random.rand(output_dim)

    def forward(self, x):
        self.x = np.array(x)
        self.input_shape = self.x.shape
        out = np.dot(x, self.weight) + self.bias
        return out

    # def backward(self, eta, lr):
    #     N, _ = eta.shape
    #     next_eta = np.dot(eta, self.weight.T)
    #     self.weight_grad = np.reshape(next_eta, self.input_shape)
    #     # x: (N, feature_num) => (N, output_dim, feature_num)
    #     x = self.x.repeat(self.output_dim, axis=0).reshape((N, self.output_dim, -1))
    #     # eta: (N, output_dim)
    #     # delta_W_grad: (N, output_dim, feature_num)
    #     self.delta_W_grad = x * eta.reshape((N, -1, 1))
    #     # delta_W_grad: (output_dim, feature_num)
    #     self.delta_W_grad = np.sum(self.delta_W_grad, axis=0) / N
    #     # delta_b_grad: (output_dim, )
    #     self.delta_b_grad = np.sum(eta, axis=0) / N
    #     self.weight -= lr * self.delta_W_grad.T
    #     self.bias -= lr * self.delta_b_grad
    #     return self.weight_grad
    def backward(self, d_output, learning_rate):
        d_input = np.dot(d_output, self.weight.T)
        self.d_weights = np.dot(self.x.T, d_output)
        self.d_biases = np.sum(d_output, axis=0)
        self.weight -= learning_rate * self.d_weights
        self.bias -= learning_rate * self.d_biases
        return d_input

# if __name__ == "__main__":
#     input = np.array([[1,2,3]])
#     fc_layer = FullyConnection(3, 2)
#     out = fc_layer.forward(input)
#     print(out)
#     W_grad = fc_layer.backward(np.array([[0.5,0.2]]), 0.1)
#     print(W_grad)