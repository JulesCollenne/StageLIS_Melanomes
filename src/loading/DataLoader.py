import numpy as np
import sklearn
from sklearn.preprocessing import MinMaxScaler

import cfg


class DataLoader:
    def __init__(self, featuresNames, featuresPath=cfg.FEATURES_PATH,
                 scaler=sklearn.preprocessing.MinMaxScaler):
        self.names = featuresNames
        self.path = featuresPath
        self.scaler = scaler

    def load(self, normalize=True, centering=False):
        featuresPath = self.path
        featuresNames = self.names
        scaler = self.scaler()
        X_trainL = []
        X_testL = []
        y_train = np.loadtxt(featuresPath + 'y_train.txt')
        y_test = np.loadtxt(featuresPath + 'y_test.txt')
        if type(featuresNames) != str:
            for feature in featuresNames:
                xtrain = np.loadtxt(featuresPath + 'X_train_' + feature + '.txt')
                xtest = np.loadtxt(featuresPath + 'X_test_' + feature + '.txt')
                if normalize:
                    scaler.fit(xtrain)
                    xtrain = scaler.transform(xtrain)
                    xtest = scaler.transform(xtest)
                if centering:
                    xtrain -= 0.5
                    xtest -= 0.5
                X_trainL.append(xtrain)
                X_testL.append(xtest)
            X_train = np.concatenate([elt for elt in X_trainL], axis=1)
            X_test = np.concatenate([elt for elt in X_testL], axis=1)
        else:
            datas = np.loadtxt(featuresPath + 'X_train_' + featuresNames + '.txt')
            scaler.fit(datas)
            X_train = scaler.transform(datas)
            X_test = scaler.transform(np.loadtxt(featuresPath + 'X_test_' + featuresNames + '.txt'))
        return X_train, y_train, X_test, y_test
