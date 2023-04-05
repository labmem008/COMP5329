from parameter import Parameter
import numpy as np

class BatchNorm:
    
    def __init__(self, shape, is_test=False):
        # Initialize gamma and beta parameters with random values
        self.gamma = Parameter(np.random.uniform(0.9, 1.1, shape))
        self.beta = Parameter(np.random.uniform(-0.1, 0.1, shape))
        # Set epsilon value for numerical stability
        self.eps = 1e-8
        # Set flag to identify training or inference mode
        self.is_test = is_test
        # Set coefficient for moving average calculation
        self.coe = 0.02
        # Initialize the overall variance and mean parameters with zero values
        self.overall_var = Parameter(np.zeros(shape))
        self.overall_ave = Parameter(np.zeros(shape))
    
    def forward(self, x):
        if not self.is_test:
            # Calculate the mean, variance, and standard deviation of the input data
            sample_ave = x.mean(axis=0)
            sample_var = x.var(axis=0)
            sample_std = np.sqrt(sample_var + self.eps)
            # Update the moving average of variance and mean
            self.overall_ave.data = (1 - self.coe) * self.overall_ave.data + self.coe * sample_ave
            self.overall_var.data = (1 - self.coe) * self.overall_var.data + self.coe * sample_var
            # Normalize the input data
            sample_diff = x - sample_ave
            self.normalized = sample_diff / sample_std
            # Scale the normalized data with gamma
            self.gamma_s = self.gamma.data / sample_std
        # Return the normalized and scaled input data after adding bias with beta
        return self.gamma.data * self.normalized + self.beta.data
    
    # Define the backward pass for batch normalization
    def backward(self, eta):         
        # Calculate the gradients of gamma and beta
        self.beta.grad = eta.mean(axis=0)
        self.gamma.grad = (eta * self.normalized).mean(axis=0)
        # Calculate the backward pass
        return self.gamma_s * (eta - self.normalized * self.gamma.grad - self.beta.grad)
