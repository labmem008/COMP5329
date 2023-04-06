import numpy as np
from linear import Linear
from activations import *
from batchnorm import BatchNorm
from dropout import Dropout
from inputnorm import InputNorm
# Define a neural network model class
class Model:
    def __init__(self, layer_configs):
        self.layers = []
        self.params = []
        self.dropout_ix = []
        self.norm_ix=[]
        # Add layers to the neural network
        for config in layer_configs:
            try:
                # Instantiate layer objects
                hyperparam = config.get('hyperparam', None)
                if hyperparam is not None:
                    # create layer by using layer name -> eval(layer_name:str)
                    layer = eval(f"{config['name']}(**hyperparam)")
                else:
                    layer = eval(f"{config['name']}()")
                self.layers.append(layer)
                # Identify layers that require dropout
                if config['name'] == 'Dropout':
                    self.dropout_ix.append(1)
                else:
                    self.dropout_ix.append(0)
                # Identify layers that require input normalization
                if config['name'] == 'InputNorm':
                    self.norm_ix.append(1)
                else:
                    self.norm_ix.append(0)
                # Add layer parameters to the list of trainable parameters
                if hasattr(layer, 'W'): self.params.append(layer.W)
                if hasattr(layer, 'b'): self.params.append(layer.b)
            except Exception as e:
                print(e)
                continue
        # Set the index of the dropout and normalization layers
        if sum(self.dropout_ix) == 0:
            self.dropout_ix = -1
        else:
            self.dropout_ix = np.argmax(np.array(self.dropout_ix))
        if sum(self.norm_ix) == 0:
            self.norm_ix = -1
        else:
            self.norm_ix = np.argmax(np.array(self.norm_ix))
    
    # Set dropout and normalization layers to training mode
    def train_mode(self):
        if self.dropout_ix != -1:
            self.layers[self.dropout_ix].is_test = False
        if self.norm_ix != -1:
            self.layers[self.norm_ix].is_test = False
    
    # Set dropout and normalization layers to testing mode
    def test_mode(self):
        if self.dropout_ix != -1:
            self.layers[self.dropout_ix].is_test = True
        if self.norm_ix != -1:
            self.layers[self.norm_ix].is_test = True
    
    # Forward pass through the neural network
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    # Backward pass through the neural network
    def backward(self, eta):
        for layer in self.layers[::-1]:
            eta = layer.backward(eta)
        return eta
