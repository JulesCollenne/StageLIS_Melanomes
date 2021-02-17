import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications import EfficientNetB5
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.models import Sequential
import cfg


def get_img_augmentation():
    img_augmentation = Sequential(
        [
            preprocessing.RandomRotation(factor=0.15),
            preprocessing.RandomTranslation(height_factor=0.1, width_factor=0.1),
            preprocessing.RandomFlip(),
            preprocessing.RandomContrast(factor=0.1),
        ],
        name="img_augmentation",
    )
    return img_augmentation


def build_model(num_classes):
    img_augmentation = get_img_augmentation()

    inputs = layers.Input(shape=(cfg.IMG_SIZE, cfg.IMG_SIZE, 3))
    x = img_augmentation(inputs)
    #model = EfficientNetB7(include_top=False, input_tensor=x, weights="imagenet")
    model = EfficientNetB0(include_top=False, input_tensor=x, weights="imagenet")

    # Freeze the pretrained weights
    model.trainable = False

    # Rebuild top
    x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = layers.BatchNormalization()(x)

    top_dropout_rate = 0.2
    x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="pred")(x)

    # Compile
    model = tf.keras.Model(inputs, outputs, name="EfficientNet")
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model
