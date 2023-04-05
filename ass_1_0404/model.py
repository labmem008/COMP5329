import numpy as np
from linear import Linear
from activations import *
from batchnorm import BatchNorm
from dropout import Dropout
from inputnorm import InputNorm

class Model:
    def __init__(self, layer_configs):
        self.layers = []
        self.params = []
        for config in layer_configs:
            try:
                hyperparam = config.get('hyperparam', None)
                if hyperparam is not None:
                    layer = eval(f"{config['name']}(**hyperparam)")
                else:
                    layer = eval(f"{config['name']}()")
                self.layers.append(layer)
                if hasattr(layer, 'W'): self.params.append(layer.W)
                if hasattr(layer, 'b'): self.params.append(layer.b)
            except Exception as e:
                print(e)
                continue
    
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, eta):
        for layer in self.layers[::-1]:
            eta = layer.backward(eta)
        return eta
            


    