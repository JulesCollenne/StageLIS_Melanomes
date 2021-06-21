import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications.efficientnet import *
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.models import Sequential
import numpy as np

import cfg
from dataset import get_datasets
from metrics_utils import visu_results

img_augmentation = Sequential(
    [
        preprocessing.RandomRotation(factor=0.8),
        preprocessing.RandomTranslation(height_factor=0.1, width_factor=0.1),
        preprocessing.RandomFlip(),
        preprocessing.RandomContrast(factor=0.5),
    ],
    name="img_augmentation",
)

name2size = {
    'b0': 224,
    'b1': 240,
    'b2': 260,
    'b3': 300,
    'b4': 380,
    'b5': 456,
    'b6': 528,
    'b7': 600,
}

name2model = {
    'b0': EfficientNetB0,
    'b1': EfficientNetB1,
    'b2': EfficientNetB2,
    'b3': EfficientNetB3,
    'b4': EfficientNetB4,
    'b5': EfficientNetB5,
    'b6': EfficientNetB6,
    'b7': EfficientNetB7,
}

names = ('b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7')


def get_img_augmentation(size):
    return Sequential(
        [
            preprocessing.Resizing(size, size),
            preprocessing.RandomRotation(factor=0.8),
            preprocessing.RandomTranslation(height_factor=0.1, width_factor=0.1),
            preprocessing.RandomFlip(),
            preprocessing.RandomContrast(factor=0.5),
        ],
        name="img_augmentation",
    )


def get_efficientnet(model_name, data_aug=True, weights="imagenet"):
    size = name2size[model_name]

    inputs = layers.Input(shape=(None, None, 3))
    if data_aug:
        x = get_img_augmentation(size)(inputs)
        model = name2model[model_name](include_top=False, input_tensor=x, weights=weights)
    else:
        model = name2model[model_name](include_top=False, input_tensor=inputs, weights=weights)

    # Freeze the pretrained weights
    if weights is not None:
        model.trainable = False

    # Rebuild top
    x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = layers.BatchNormalization()(x)

    top_dropout_rate = 0.2
    x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = layers.Dense(2, activation="softmax", name="pred")(x)

    # Compile
    model = tf.keras.Model(inputs, outputs, name="EfficientNet")
    model.compile(
        optimizer='adam', loss="binary_crossentropy", metrics='AUC'
    )
    return model


def CNN_predict(model, dataset):
    y_true = []
    y_pred = []
    for X, y in dataset:
        y_true += y.numpy().tolist()
        y_pred += model(X).numpy().tolist()
    return y_true, y_pred


if __name__ == "__main__":
    print("Getting dataset...")
    train_ds, val_ds, test_ds = get_datasets()
    sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
    print("Loading model...")
    b2 = get_efficientnet('b2')
    b2.load_weights(cfg.WEIGHTS_PATH + 'b2_fine.h5')
    print("Predictions...")
    y_testCNN, y_predCNN = CNN_predict(b2, test_ds)
    print("Done!")
    visu_results(y_testCNN, np.asarray(y_predCNN), confidence=np.asarray(y_predCNN))
