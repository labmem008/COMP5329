import os
import sys
import time
import numpy as np
from tqdm import tqdm 
from optimizer import *
from model import Model
from activations import Softmax
from cross_entropy import CrossEntropy
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


softmax = Softmax()

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


def train(net, loss_fn, train_data_path, train_label_path, test_data_path, test_label_path, batch_size, optimizer, load_file, save_as, epoches=1, patience=5, save_result=False, title=None):
    best_loss = 1e3
    best_model = net
    X, Y = load_data(train_data_path, train_label_path, transform=True)
    test_X, test_Y = load_data(test_data_path, test_label_path, transform=True)
    train_X, dev_X, train_Y, dev_Y = train_test_split(X, Y, test_size=0.15, random_state=42)
    data_size = train_X.shape[0]
    train_accs, train_losses = [], []
    dev_accs, dev_losses = [], []
    with tqdm(total=epoches, desc=f'Epoch0') as pbar:
        for epoch in range(epoches):
            net.train_mode()
            i = 0
            batch_losses = []
            batch_acces = []
            while i <= data_size - batch_size:
                x = train_X[i:i+batch_size]
                y = train_Y[i:i+batch_size]
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
            net.test_mode()
            dev_acc, dev_loss, dev_rpt = test(net, loss_fn, dev_X, dev_Y)
            dev_acc = np.average(dev_acc)
            train_accs.append(epoch_acc)
            train_losses.append(epoch_loss)
            dev_accs.append(dev_acc)
            dev_losses.append(dev_loss)
            pbar.set_description(f'Epoch{epoch}, epoch_loss: {round(epoch_loss, 4)}, acc: {round(epoch_acc, 4)*100}%, dev_loss: {round(dev_loss, 4)}, dev_acc: {round(dev_acc, 4)*100}%')
            pbar.update(1)
            # if epoch%5==0:
            #     print('dev report', dev_rpt, sep='\n')
            # early_stopping
            if dev_loss < best_loss:
                best_loss = min(dev_loss, best_loss)
                best_model = net
                p = 0
            else:
                p += 1
            if p > patience:
                break
        net.test_mode()
        start_time = time.time()
        test_acc, test_loss, test_rpt = test(best_model, loss_fn, test_X, test_Y)
        run_time = time.time() - start_time
        test_acc = np.average(test_acc)
        print(f'epoch_loss: {round(test_loss, 4)}, acc: {round(test_acc, 4)*100}%\n{test_rpt}')
        if save_as is not None: save(net.parameters, save_as)
        if save_result:
            train_accs = '|'.join([str(i) for i in train_accs])
            train_losses = '|'.join([str(i) for i in train_losses])
            dev_accs = '|'.join([str(i) for i in dev_accs])
            dev_losses = '|'.join([str(i) for i in dev_losses])
            test_loss = str(test_loss)
            test_acc = str(test_acc)
            run_time = str(run_time)
            line = ', '.join([title, train_accs, train_losses, dev_accs, dev_losses, test_loss, test_acc, run_time])
            f = open('ass_1_0404/hyperparam_result.txt', 'a')
            f.write(line+'\n')
            f.close()


    
def test(net, loss_fn, data, label):
    output = net.forward(data)
    prediction = softmax.forward(output)
    acc, loss = loss_fn(output, label)
    rpt = classification_report(np.argmax(label, axis=1), np.argmax(prediction, axis=1))
    return acc, loss, rpt




if __name__ == "__main__": 
    layers = [
        {'name': 'InputNorm', 'hyperparam': {'shape': 128}},

        {'name': 'Linear', 'hyperparam': {'in_dim': 128, 'out_dim': 128}},
        {'name': 'BatchNorm', 'hyperparam': {'shape': 128}},
        {'name': 'Relu'},
        
        {'name': 'Linear', 'hyperparam': {'in_dim': 128, 'out_dim': 256}},
        {'name': 'BatchNorm', 'hyperparam': {'shape': 256}},
        {'name': 'Relu'},

        # {'name': 'Linear', 'hyperparam': {'in_dim': 256, 'out_dim': 768}},
        # {'name': 'BatchNorm', 'hyperparam': {'shape': 768}},
        # {'name': 'Relu'},

        # {'name': 'Linear', 'hyperparam': {'in_dim': 768, 'out_dim': 256}},
        # {'name': 'BatchNorm', 'hyperparam': {'shape': 256}},
        {'name': 'Relu'},

        {'name': 'Linear', 'hyperparam': {'in_dim': 256, 'out_dim': 128}},
        {'name': 'BatchNorm', 'hyperparam': {'shape': 128}},
        {'name': 'Relu'},

        {'name': 'Linear', 'hyperparam': {'in_dim': 128, 'out_dim': 16}},
        {'name': 'BatchNorm', 'hyperparam': {'shape': 16}},
        {'name': 'Relu'},
        {'name': 'Dropout', 'hyperparam': {'drop_rate': 0.5}},

        {'name': 'Linear', 'hyperparam': {'in_dim': 16, 'out_dim': 10}},
    ]
    loss_fn = CrossEntropy()
    net = Model(layers)
    lr = 0.1
    batch_size = 64
    optimizer = Adam(net.params, lr, decay=0)
    train_data_path = './data/train_data.npy'
    train_label_path = './data/train_label.npy'
    test_data_path = './data/test_data.npy'
    test_label_path = './data/test_label.npy'
    data_files = (train_data_path, train_label_path, test_data_path, test_label_path)
    train(net, loss_fn, *data_files, batch_size, optimizer, None, None, epoches=100, patience=10, save_result=True, title='Baseline-Adam-batchsize:64')
