import numpy as np
# from sklearn.preprocessing import OneHotEncoder


def data_reader(X_path, y_path):
    X = np.load(open(X_path, 'rb'))
    y = np.load(open(y_path, 'rb'))
    return X, y

def data_loader(X, y, batch_size=32):
    # ohe = OneHotEncoder()
    # y = ohe.fit_transform(y).toarray()
    X = [X[i: i+batch_size]for i in range(0, len(X), batch_size)]
    y = [y[i: i+batch_size]for i in range(0, len(y), batch_size)]
    dataset = [[b_x, b_y] for b_x, b_y in zip(X, y)]
    return dataset