import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import layers

import cfg
from CNN.models import get_efficientnet, CNN_predict
from handcrafted_models.models import get_HCANN
from loading.load_features import get_features
from metrics.metrics_utils import plot_hist, visu_results
from multimodal.hybrid_loader import CustomDataGen
from multimodal.late_fusion import load_test_set_multimodal
import numpy as np


def get_hybrid_model(cnn, hcnn):
    # cnn.pop()
    # hcnn.pop()
    cnn.trainable = False
    hcnn.trainable = False

    # combinedInput = concatenate([cnn.output, hcnn.output])
    combinedInput = concatenate([cnn.layers[-2].output, hcnn.layers[-2].output])

    combinedInput = Dropout(0.1)(combinedInput)
    x = Dense(32, activation="relu")(combinedInput)
    x = Dropout(0.1)(x)
    x = Dense(16, activation="relu")(x)
    x = Dropout(0.1)(x)
    outputs = layers.Dense(2, activation="softmax", name="predFinal")(x)

    # Compile
    model = tf.keras.Model(inputs=[cnn.inputs, hcnn.inputs], outputs=outputs, name="HybridFusion")
    model.compile(
        optimizer='adam', loss="binary_crossentropy", metrics='AUC'
    )
    return model


train_df = pd.read_csv('/content/drive/MyDrive/Stage_LIS/train_files.csv')
test_df = pd.read_csv('/content/drive/MyDrive/Stage_LIS/test_files.csv')

# -------- ANN -------

X_trainANN, y_trainANN, X_testANN, y_testANN = get_features()
hcann = get_HCANN()
hcann.load_weights(cfg.WEIGHTS_PATH + 'hcann.h5')
y_predANN = hcann.predict(X_testANN)

# ------- CNN ------

X_testCNN = load_test_set_multimodal()
b2 = get_efficientnet('b2')
b2.load_weights(cfg.WEIGHTS_PATH + 'b2_fine.h5')
y_predCNN = b2.predict(X_testCNN)

# ----- Hybrid fusion -----

traingen = CustomDataGen(train_df, X_trainANN, val_split=0.2,
                         batch_size=32)

valgen = CustomDataGen(train_df, X_trainANN, val_split=0.2, is_val=True,
                       batch_size=32)

testgen = CustomDataGen(test_df, X_testANN,
                        batch_size=32)

hybrid = get_hybrid_model(b2, hcann)

# Training

epochs = 5
batch_size = 32
class_weight = {0: 4.59, 1: 1}

hist = hybrid.fit(traingen, validation_data=valgen, epochs=epochs, class_weight=class_weight, batch_size=batch_size)

y_test, predictions = CNN_predict(hybrid, testgen)

plot_hist(hist, metrics='auc')
visu_results(np.asarray(y_test), np.asarray(predictions), confidence=np.asarray(predictions))
