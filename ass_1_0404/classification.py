import os
import sys
import numpy as np
from tqdm import tqdm 
from optimizer import *
from model import Model
from cross_entropy import CrossEntropy
from sklearn.preprocessing import OneHotEncoder


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
    ohe = OneHotEncoder()
    X = np.load(open(train_data_path, 'rb'))
    Y = np.load(open(train_label_path, 'rb'))
    if transform:
        X = X.reshape(len(X), -1)
    Y = ohe.fit_transform(Y).toarray()
    return X, Y


def train(net, loss_fn, train_data_path, train_label_path, batch_size, optimizer, load_file, save_as, times=1, retrain=False):
    X, Y = load_data(train_data_path, train_label_path, transform=True)
    data_size = X.shape[0]
    if not retrain and os.path.isfile(load_file): load(net.parameters, load_file)
    with tqdm(total=times, desc=f'Epoch0') as pbar:
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
                # print(net.params)
                optimizer.update()
                batch_losses.append(batch_loss)
                batch_acces.extend(batch_acc)
            epoch_loss = np.average(batch_losses)
            epoch_acc = np.average(batch_acces)
            pbar.set_description(f'Epoch{loop}, epoch_loss: {round(epoch_loss, 4)}, acc: {round(epoch_acc, 4)*100}%')
            pbar.update(1)
        if save_as is not None: save(net.parameters, save_as)
    

if __name__ == "__main__": 
    layers = [
        {'name': 'InputNorm', 'hyperparam': {'shape': 128}},

        {'name': 'Linear', 'hyperparam': {'in_dim': 128, 'out_dim': 256}},
        {'name': 'BatchNorm', 'hyperparam': {'shape': 256}},
        {'name': 'Gelu'},
        {'name': 'Dropout', 'hyperparam': {'drop_rate': 0.4}},

        {'name': 'Linear', 'hyperparam': {'in_dim': 256, 'out_dim': 512}},
        {'name': 'BatchNorm', 'hyperparam': {'shape': 512}},
        {'name': 'Gelu'},
        # {'name': 'Dropout', 'hyperparam': {'drop_rate': 0.2}},

        {'name': 'Linear', 'hyperparam': {'in_dim': 512, 'out_dim': 256}},
        {'name': 'BatchNorm', 'hyperparam': {'shape': 256}},
        {'name': 'Gelu'},
        # {'name': 'Dropout', 'hyperparam': {'drop_rate': 0.2}},

        {'name': 'Linear', 'hyperparam': {'in_dim': 256, 'out_dim': 128}},
        {'name': 'BatchNorm', 'hyperparam': {'shape': 128}},
        {'name': 'Gelu'},
        # {'name': 'Dropout', 'hyperparam': {'drop_rate': 0.2}},

        {'name': 'Linear', 'hyperparam': {'in_dim': 128, 'out_dim': 16}},
        {'name': 'BatchNorm', 'hyperparam': {'shape': 16}},
        {'name': 'Gelu'},
        # {'name': 'Dropout', 'hyperparam': {'drop_rate': 0.2}},

        {'name': 'Linear', 'hyperparam': {'in_dim': 16, 'out_dim': 10}},
    ]
    loss_fn = CrossEntropy()
    net = Model(layers)
    lr = 0.1
    batch_size = 512
    optimizer = SGD(net.params, lr, decay=0)
    train_data_path = './data/train_data.npy'
    train_label_path = './data/train_label.npy'
    train(net, loss_fn, train_data_path, train_label_path, batch_size, optimizer, None, None, times=50, retrain=True)
