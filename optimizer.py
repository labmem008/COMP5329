import numpy as np


# Stochastic Gradient Descent optimizer
class SGD(object):
    def __init__(self, parameters, lr, decay=0):
        self.parameters = parameters
        self.lr = lr
        self.decay_rate = 1.0 - decay

    # Update the model parameters using SGD
    def update(self):
        for p in self.parameters:
            p.data *= self.decay_rate
            p.data -= self.lr * p.grad


# Momentum optimizer
class Momentum(object):
    def __init__(self, parameters, lr, decay=0, beta=0.9):
        self.lr = lr
        self.decay_rate = 1.0 - decay
        self.beta = beta
        self.parameters = parameters
        self.accmulated_grads = [np.zeros(p.data.shape) for p in self.parameters]

    # Update the model parameters using momentum
    def update(self):
        for p, grad in zip(self.parameters, self.accmulated_grads):
            p.data *= self.decay_rate
            grad = self.beta * grad + (1 - self.beta) * p.grad
            p.data -= self.lr * grad

# Adam optimizer for neural network training
# Inputs:
#     parameters: model parameters to be optimized
#     lr: learning rate
#     decay: learning rate decay
#     beta1: decay rate for the moving average of the gradient
#     beta2: decay rate for the moving average of the squared gradient
#     eps: small value to avoid division by zero
class Adam(object):
    def __init__(self, parameters, lr, decay=0, beta1=0.9, beta2=0.999, eps=1e-8):
        self.beta1 = beta1
        self.beta2 = beta2
        self.accumulated_beta1 = 1
        self.accumulated_beta2 = 1
        self.lr = lr
        self.decay_rate = 1.0 - decay
        self.eps = eps
        self.parameters = parameters
        self.accumulated_grad_mom = [np.zeros(p.data.shape) for p in self.parameters]
        self.accumulated_grad_rms = [np.zeros(p.data.shape) for p in self.parameters]

    # Update model parameters using Adam algorithm
    def update(self):
        self.accumulated_beta1 *= self.beta1
        self.accumulated_beta2 *= self.beta2
        lr = self.lr * ((1 - self.accumulated_beta2)**0.5) / (1 - self.accumulated_beta1)
        for p, grad_mom, grad_rms in zip(self.parameters, self.accumulated_grad_mom, self.accumulated_grad_rms):
            # Decay model parameters
            p.data *= self.decay_rate
            # Update moving averages of the gradient
            np.copyto(grad_mom, self.beta1 * grad_mom + (1 - self.beta1) * p.grad)
            # Update moving averages of the squared gradient
            np.copyto(grad_rms, self.beta2 * grad_rms + (1 - self.beta2) * np.power(p.grad, 2))
            # Update model parameters using the computed values
            p.data -= lr * grad_mom / (np.sqrt(grad_rms) + self.eps)
