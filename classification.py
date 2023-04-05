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



# Define a Softmax object
softmax = Softmax()

# Define a function to save model parameters as a NumPy binary file
def save(parameters, save_as):
    dic = {}
    for i in range(len(parameters)):
        dic[str(i)] = parameters[i].data
    np.savez(save_as, **dic)
    
# Define a function to load model parameters from a NumPy binary file
def load(parameters, file):
    params = np.load(file)
    for i in range(len(parameters)):
        parameters[i].data = params[str(i)]
        
# Define a function to load training data and labels
def load_data(train_data_path, train_label_path, transform=False):
    # Initialize a OneHotEncoder object
    ohe = OneHotEncoder()
    # Load the training data and labels using NumPy
    X = np.load(open(train_data_path, 'rb'))
    Y = np.load(open(train_label_path, 'rb'))
    # Reshape the training data if required
    if transform:
        X = X.reshape(len(X), -1)
    # Convert the labels to one-hot encoding using OneHotEncoder
    Y = ohe.fit_transform(Y).toarray()
    return X, Y



# Train a neural network model
# Inputs:
#     net: neural network model
#     loss_fn: loss function
#     train_data_path: path to training data
#     train_label_path: path to training labels
#     test_data_path: path to testing data
#     test_label_path: path to testing labels
#     batch_size: batch size for training
#     optimizer: optimization algorithm
#     load_file: file name for loading trained model
#     save_as: file name for saving trained model
#     epoches: number of epochs for training
#     patience: number of epochs to wait for improvement in validation loss
#     save_result: whether to save training results
#     title: title for the training progress bar
def train(net, loss_fn, train_data_path, train_label_path, test_data_path, test_label_path, batch_size, optimizer, load_file, save_as, epoches=1, patience=5, save_result=False, title=None):
    best_loss = 1e3
    best_model = net
    # Load training and testing data
    X, Y = load_data(train_data_path, train_label_path, transform=True)
    test_X, test_Y = load_data(test_data_path, test_label_path, transform=True)
    # Split training data into training and development sets
    train_X, dev_X, train_Y, dev_Y = train_test_split(X, Y, test_size=0.15, random_state=42)
    data_size = train_X.shape[0]
    train_accs, train_losses = [], []
    dev_accs, dev_losses = [], []
    # Train the model for the specified number of epochs
    with tqdm(total=epoches, desc=f'Epoch0') as pbar:
        for epoch in range(epoches):
            net.train_mode()
            i = 0
            batch_losses = []
            batch_acces = []
            # Train the model using batches
            while i <= data_size - batch_size:
                x = train_X[i:i+batch_size]
                y = train_Y[i:i+batch_size]
                i += batch_size
                # Forward and backward propagation
                output = net.forward(x)
                batch_acc, batch_loss = loss_fn(output, y)
                eta = loss_fn.gradient()
                net.backward(eta)
                optimizer.update()
                batch_losses.append(batch_loss)
                batch_acces.extend(batch_acc)
            epoch_loss = np.average(batch_losses)
            epoch_acc = np.average(batch_acces)
            net.test_mode()
            # Evaluate the model using the development set
            dev_acc, dev_loss, dev_rpt = test(net, loss_fn, dev_X, dev_Y)
            dev_acc = np.average(dev_acc)
            # Record training and development accuracy and loss
            train_accs.append(epoch_acc)
            train_losses.append(epoch_loss)
            dev_accs.append(dev_acc)
            dev_losses.append(dev_loss)
            # Update progress bar
            pbar.set_description(f'Epoch{epoch}, epoch_loss: {round(epoch_loss, 4)}, acc: {round(epoch_acc, 4)*100}%, dev_loss: {round(dev_loss, 4)}, dev_acc: {round(dev_acc, 4)*100}%')
            pbar.update(1)

            
            if dev_loss < best_loss:
                # Update the best loss and model
                best_loss = min(dev_loss, best_loss)
                best_model = net
                p = 0
            else:
                p += 1 # Increment the patience counter
            if p > patience:
                # Early stopping if the validation loss does not improve for a certain number of epochs
                print('Early Stop!')
                break
        # Evaluate the model on the testing set
        net.test_mode()
        start_time = time.time()
        test_acc, test_loss, test_rpt = test(best_model, loss_fn, test_X, test_Y)
        run_time = time.time() - start_time
        test_acc = np.average(test_acc)
        print(f'test_loss: {round(test_loss, 4)}, test_acc: {round(test_acc, 4)*100}%\n{test_rpt}')
        # Save the trained model and training results
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
            f = open('ass_1_0404/module_result.txt', 'a')
            f.write(line+'\n')
            f.close()


    
# Test the trained neural network model
# Inputs:
#     net: neural network model
#     loss_fn: loss function
#     data: testing data
#     label: testing labels
# Outputs:
#     acc: accuracy of the model
#     loss: loss of the model
#     rpt: classification report of the model
def test(net, loss_fn, data, label):
    output = net.forward(data)
    prediction = softmax.forward(output)
    acc, loss = loss_fn(output, label)
    rpt = classification_report(np.argmax(label, axis=1), np.argmax(prediction, axis=1), zero_division=1)
    return acc, loss, rpt

if __name__ == "__main__": 
    # Define neural network layers and loss function
    layers = [
        {'name': 'InputNorm', 'hyperparam': {'shape': 128}},
        {'name': 'Linear', 'hyperparam': {'in_dim': 128, 'out_dim': 256}},
        {'name': 'BatchNorm', 'hyperparam': {'shape': 256}},
        {'name': 'Relu'},
        {'name': 'Linear', 'hyperparam': {'in_dim': 256, 'out_dim': 128}},
        {'name': 'BatchNorm', 'hyperparam': {'shape': 128}},
        {'name': 'Relu'},
        {'name': 'Linear', 'hyperparam': {'in_dim': 128, 'out_dim': 16}},
        {'name': 'BatchNorm', 'hyperparam': {'shape': 16}},
        {'name': 'Relu'},
        {'name': 'Linear', 'hyperparam': {'in_dim': 16, 'out_dim': 10}},
    ]
    loss_fn = CrossEntropy()
    # Create a neural network model
    net = Model(layers)
    # Define optimization algorithm and parameters
    lr = 0.1
    batch_size = 256
    optimizer = Adam(net.params, lr, decay=1e-5)
    # Load training and testing data
    train_data_path = 'Multilayer-Neural-Network-final/data/train_data.npy'
    train_label_path = 'Multilayer-Neural-Network-final/data/train_label.npy'
    test_data_path = 'Multilayer-Neural-Network-final/data/test_data.npy'
    test_label_path = 'Multilayer-Neural-Network-final/data/test_label.npy'
    data_files = (train_data_path, train_label_path, test_data_path, test_label_path)
    # Train the model
    train(net, loss_fn, *data_files, batch_size, optimizer, None, None, epoches=100, patience=10, save_result=False, title='Baseline-Momentum')
