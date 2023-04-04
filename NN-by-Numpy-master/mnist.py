import numpy as np
import package
import package.optim as optim
import os
import sys


def save(parameters, save_as):
    dic = {}
    for i in range(len(parameters)):
        dic[str(i)] = parameters[i].data
    np.savez(save_as, **dic)
    
def load(parameters, file):
    params = np.load(file)
    for i in range(len(parameters)):
        parameters[i].data = params[str(i)]


def load_data(train_data_path, train_label_path, transform=False):
    X = np.load(open(train_data_path, 'rb'))
    Y = np.load(open(train_label_path, 'rb'))
    if transform:
        X = X.reshape(len(X), -1)
    return X, Y


def train(net, loss_fn, train_data_path, train_label_path, batch_size, optimizer, load_file, save_as, times=1, retrain=False):
    X, Y = load_data(train_data_path, train_label_path, transform=True)
    data_size = X.shape[0]
    if not retrain and os.path.isfile(load_file): load(net.parameters, load_file)
    for loop in range(times):
        i = 0
        batch_losses = []
        batch_acces = []
        while i <= data_size - batch_size:
            x = X[i:i+batch_size]
            y = Y[i:i+batch_size]
            i += batch_size

            output = net.forward(x)
            batch_acc, batch_loss = loss_fn(output, y)
            eta = loss_fn.gradient()
            net.backward(eta)
            optimizer.update()
            # if i % 50 == 0:
            #     print("loop: %d, batch: %5d, batch acc: %2.1f, batch loss: %.2f" % \
            #         (loop, i, batch_acc*100, batch_loss))
            batch_losses.append(batch_loss)
            batch_acces.append(batch_acc)
        epoch_loss = np.average(batch_losses)
        epoch_acc = np.average(batch_acces)
        print(f"epoch: {loop}, loss: {epoch_loss}, acc: {epoch_acc}")
        pass
    if save_as is not None: save(net.parameters, save_as)
    

if __name__ == "__main__": 
    layers = [
        {'type': 'batchnorm', 'shape': 128, 'requires_grad': False, 'affine': False},
        {'type': 'linear', 'shape': (128, 64)},
        {'type': 'batchnorm', 'shape': 64},
        {'type': 'relu'},
        {'type': 'linear', 'shape': (64, 32)},
        {'type': 'batchnorm', 'shape': 32},
        {'type': 'relu'},
        {'type': 'linear', 'shape': (32, 10)}
    ]
    loss_fn = package.CrossEntropyLoss()
    net = package.Net(layers)
    lr = 0.001
    batch_size = 128
    optimizer = optim.Adam(net.parameters, lr)
    train_data_path = './data/train_data.npy'
    train_label_path = './data/train_label.npy'
    # param_file = './data/param.npz'
    train(net, loss_fn, train_data_path, train_label_path, batch_size, optimizer, None, None, times=50, retrain=True)
