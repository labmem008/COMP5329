from parameter import Parameter
import numpy as np
# Normalize input data
# Inputs:
#     shape: shape of input data
#     is_test: whether the input data is for testing
class InputNorm:
    def __init__(self, shape, is_test=False):
        self.eps = 1e-8
        self.is_test = is_test
        self.coe = 0.02
        # Initialize variance and mean for normalization
        self.overall_var = Parameter(np.zeros(shape))
        self.overall_ave = Parameter(np.zeros(shape))

    # Normalize the input data
    def forward(self, x):
        if self.is_test:
            # For testing data, use the overall mean and variance for normalization
            sample_ave = self.overall_ave.data
            sample_std = np.sqrt(self.overall_var.data)
        else:
            # For training data, calculate the mean and variance for normalization
            sample_ave = x.mean(axis=0)
            sample_var = x.var(axis=0)
            sample_std = np.sqrt(sample_var + self.eps)
            # Update overall variance and mean using a moving average
            self.overall_ave.data = (1 - self.coe) * self.overall_ave.data + self.coe * sample_ave
            self.overall_var.data = (1 - self.coe) * self.overall_var.data + self.coe * sample_var
        # Normalize the input data
        return (x - sample_ave) / sample_std

    # No backward propagation is needed for input normalization
    def backward(self,eta):
        return

