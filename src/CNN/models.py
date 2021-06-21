import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications.efficientnet import *
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.models import Sequential

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
