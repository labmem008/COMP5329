import numpy as np
from config import *
from tqdm import tqdm
from loss import CrossEntropy
from model.Model2 import MModel2 as Model
# from model.MultiLayerModel import MModel as Model
from load_data import data_loader, data_reader
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

plt.ion() #开启interactive mode 成功的关键函数
plt.figure(1)
plt.title("dev loss")
plt.xlabel("epoches")
plt.ylabel("loss")
def test(model, dataset):
    loss_fn = CrossEntropy()
    predictions, labels = [], []
    for batch in dataset:
        data, label = batch
        pred, batch_loss = model.forward(data, label)
        # pred = np.argmax(pred, axis=1)
        predictions.extend(pred)
        labels.extend(label)
    cls_rpt = classification_report(labels, np.argmax(predictions, axis=1), zero_division=1)
    # print(predictions)
    labels = np.array(labels)
    loss = loss_fn(predictions, labels)
    return model, cls_rpt, loss


def train(model, train_dataset, test_dataset, patience):
    def each_epoch_step(model, data, train=True):
        predictions, labels = [], []
        for batch in data:
            data, label = batch
            pred, batch_loss = model.forward(data, label)
            # print(e, pred[0], label[0])
            if train:
                model.backward(LR)
            # pred = np.argmax(pred, axis=1)
            predictions.extend(pred)
            labels.extend(label)
        cls_rpt = classification_report(labels, np.argmax(predictions, axis=1), zero_division=1)
        predictions = np.array(predictions)
        labels = np.array(labels)
        loss = loss_fn(predictions, labels)
        return model, cls_rpt, loss
    
    data, label = train_dataset
    train_X, dev_X, train_y, dev_y = train_test_split(data, label, test_size=0.1)
    train_data = data_loader(train_X, train_y, batch_size=BATCH_SIZE)
    dev_data = data_loader(dev_X, dev_y, batch_size=BATCH_SIZE)
    train_loss, dev_loss = 0, 0
    loss_fn = CrossEntropy()
    best_loss = 1e4
    best_model = model
    e = 0
    p = 0
    dev_losses = []
    epoch_nums = []
    with tqdm(total=EPOCH, desc=f'Epoch{e}') as pbar:
        for e in range(EPOCH):
            model, train_cls_rpt, train_loss = each_epoch_step(model, train_data)
            # print(train_cls_rpt)
            model, dev_cls_rpt, dev_loss = each_epoch_step(model, dev_data, False)
            # print(dev_cls_rpt)
            dev_losses.append(dev_loss)
            epoch_nums.append(e)
            plt.clf()
            plt.plot(epoch_nums, dev_losses)
            plt.draw()
            plt.pause(1e-20)
            plt.savefig('dev_loss.jpg')
            # early_stopping
            # if dev_loss < best_loss:
            #     best_loss = min(dev_loss, best_loss)
            #     best_model = model
            #     p = 0
            # else:
            #     p += 1
            # if p > patience:
            #     break
            pbar.set_description(f'Epoch{e}, train_loss: {round(train_loss, 4)}, dev_loss: {round(dev_loss, 4)}')
            pbar.update(1)
            
    best_model, best_cls_rpt, best_loss = test(best_model, test_dataset)
    # print('Best model loss: ', best_loss)
    # print(best_cls_rpt)
    return best_model


if __name__ == "__main__":
    model = Model()
    train_dataset = data_reader(TRAIN_DATA_PATH, TRAIN_LABEL_PATH)
    test_dataset = data_reader(TEST_DATA_PATH, TEST_LABEL_PATH)
    test_dataset = data_loader(*test_dataset, batch_size=BATCH_SIZE)
    train(model, train_dataset, test_dataset, 5)
        

    
        