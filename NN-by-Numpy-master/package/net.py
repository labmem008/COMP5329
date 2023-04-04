from .layers import *


class Net(Layer):
    def __init__(self, layer_configures):
        self.layers = []
        self.parameters = []
        for config in layer_configures:
            self.layers.append(self.createLayer(config))

    def createLayer(self, config):
        '''
        继承的子类添加自定义层可重写此方法
        '''
        return self.getDefaultLayer(config)

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, eta):
        for layer in self.layers[::-1]:
            eta = layer.backward(eta)
        return eta

    def getDefaultLayer(self, config):
        t = config['type']
        if t == 'linear':
            layer = Linear(**config)
            self.parameters.append(layer.W)
            if layer.b is not None: self.parameters.append(layer.b)
        elif t == 'relu':
            layer = Relu()
        elif t == 'gelu':
            layer = Gelu()
        elif t == 'softmax':
            layer = Softmax()
        elif t == 'dropout':
            layer = Dropout(**config)
        elif t == 'batchnorm':
            layer = BatchNorm(**config)
        else:
            raise TypeError
        return layer