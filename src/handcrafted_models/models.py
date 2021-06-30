from sklearn.preprocessing import OneHotEncoder
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.models import *

import cfg
from loading.DataLoader import DataLoader
from metrics.metrics_utils import visu_results


def get_features():
    ohe = OneHotEncoder()

    featuresNames = ['beta_quad_RGB', 'beta_quad_HSV', 'beta_quad_Lab', 'beta_quad_YCrCb',
                     'histos', 'histos_HSV', 'histos_Lab', 'histos_YCrCb',
                     'SPML_4_2', 'SPML_4_2_HSV', 'SPML_4_2_cielab', 'SPML_4_2_ycbcr',
                     'SPO', 'SP60_4']

    loader = DataLoader(featuresNames)
    X_train, y_train, X_test, y_test = loader.load(centering=True)

    ohe.fit(y_train.reshape((-1, 1)))
    y_trainOH = ohe.transform(y_train.reshape((-1, 1))).toarray()
    y_testOH = ohe.transform(y_test.reshape((-1, 1))).toarray()
    return X_train, y_trainOH, X_test, y_testOH


def get_HCANN():
    model = Sequential()
    # model.add(Dense(32, activation='relu'))
    model.add(Dense(16, input_dim=714, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['AUC'])
    return model


if __name__ == "__main__":
    X_trainANN, y_trainANN, X_testANN, y_testANN = get_features()
    hcann = get_HCANN()
    hcann.load_weights(cfg.WEIGHTS_PATH+"hcann.h5")
    y_predANN = hcann.predict(X_testANN)
    visu_results(y_testANN, y_predANN, confidence=y_predANN)
