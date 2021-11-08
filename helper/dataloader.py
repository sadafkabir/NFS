import torch
import os
import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import KFold

#dermatology
features = np.arange(1,34)
y_index = 0

class Dataloader:
    def __init__(self, file_address, batch_size=32, train_mean=None, train_std=None,
                 y_index=y_index, k=5, selected_fold=4):

        # check if the file exists!
        assert os.path.isfile(file_address), 'Cannot find the data file!'
        np.random.seed(193)
        self.address = file_address
        self.bs = batch_size
        self.epoch = 0
        self.iter = 0
        self.mean = train_mean
        self.std = train_std
        self.y_index = y_index
        self.n_features = len(features)
        self.features = features
        self.k = k

        # check if the data type is .mat
        if self.address[-4:] == '.mat':
            # load the data into a dictionary
            dict = loadmat(self.address)
            # get data from the dictionary and put it in an array
            keys = list(dict.keys())

            data = dict[keys[-1]]
            x = data

            print(x.shape)

            # get number of categories
            target = x[:, y_index].copy()

            # x0 = np.where(target == 0)[0]
            x0 = np.where(target == 1)[0]
            x0 = x[x0, ...].copy()
            # x1 = np.where(target == 1)[0]
            x1 = np.where(target == 2)[0]
            x1 = x[x1, ...].copy()

            n0 = x0.shape[0]
            n0 = n0 - n0 % 10
            n1 = x1.shape[0]
            n1 = n1 - n1 % 10

            ratio = max(n0, n1)//min(n0, n1)

            if n0>n1:
                x1 = np.tile(x1, (ratio, 1))
            elif n1>n0:
                x0 = np.tile(x0, (ratio, 1))

            x = np.concatenate((x0, x1), 0)
            # get number of categories
            target = x[:, y_index].copy()

            unique = np.unique(target)
            self.n_cats = unique.shape[0]

            # Normalization
            self.data = self.outlierRemovalAndNormalize(x)
            self.data[:, y_index] = self.data[:, y_index] > 0

            shuffle_idx = np.random.permutation(self.data.shape[0])
            shuffle_idx = np.reshape(shuffle_idx, [-1])
            self.data = self.data[shuffle_idx, :]

            # Cross validation
            kf = KFold(n_splits=k, random_state=123, shuffle=False)  # Define the split - into 2 folds
            # kf.get_n_splits(self.data)  # returns the number of splitting iterations in the cross-validator

            splits = list(kf.split(self.data))

            train_index, test_index = splits[selected_fold]
            self.X_train, self.X_test = self.data[train_index], self.data[test_index]

            print(self.X_train.shape, self.X_test.shape)
            # print(X_train)

            print(self.data.shape)


        else:
            exit('Data file is not recognized!')

    def outlierRemovalAndNormalize(self, x):
        a = np.isnan(x)
        assert a.sum() == 0, 'there is nan'

        isnone = False
        if self.mean is None:
            self.mean = x.mean(0, keepdims=True)
            isnone = True

        if self.std is None:
            self.std = x.std(0, keepdims=True)


        x_norm = (x - self.mean) / self.std

        return x_norm

    def next_train(self):
        if self.iter > self.X_train.shape[0]:
            self.epoch += 1
            self.iter = 0
            # shuffle the data
            shuffle_idx = np.random.permutation(self.X_train.shape[0])
            shuffle_idx = np.reshape(shuffle_idx, [-1])
            self.X_train = self.X_train[shuffle_idx, :]

        X_train = self.X_train[self.iter:self.iter + self.bs, :]

        Xtrain_y = X_train[:, self.y_index:self.y_index + 1]
        Xtrain_x = X_train[:, features]
        self.iter += self.bs

        Xtrainx = torch.from_numpy(Xtrain_x).type(torch.float)
        Xtrainy = torch.from_numpy(Xtrain_y).type(torch.long)
        return (Xtrainx, Xtrainy)

    def next_test(self):
        if self.iter > self.X_test.shape[0]:
            self.epoch += 1
            self.iter = 0
            # shuffle the data
            shuffle_idx = np.random.permutation(self.X_test.shape[0])
            shuffle_idx = np.reshape(shuffle_idx, [-1])
            self.X_test = self.X_test[shuffle_idx, :]

        X_test = self.X_test[self.iter:self.iter + self.bs, :]

        Xtest_y = X_test[:, self.y_index:self.y_index + 1]
        Xtest_x = X_test[:, features]
        self.iter += self.bs

        Xtestx = torch.from_numpy(Xtest_x).type(torch.float)
        Xtesty = torch.from_numpy(Xtest_y).type(torch.long)
        return (Xtestx, Xtesty)
