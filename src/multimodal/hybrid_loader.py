import tensorflow as tf
import numpy as np


class CustomDataGen(tf.keras.utils.Sequence):  # chargement des donnees fichier csv nom d images indices
    def __init__(self, df, features,
                 batch_size, val_split=0.,
                 shuffle=True, is_val=False):
        if val_split > 0:
            if is_val == False:
                self.df = df.copy()[:-int(len(df) * 0.2)]
            else:
                self.df = df.copy()[int(len(df) * 0.2):]
        else:
            self.df = df.copy()
        self.features = features
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n = len(self.df)
        self.num_classes = 2

    def on_epoch_end(self):
        if self.shuffle:
            self.df.sample(frac=1).reset_index(drop=True)

    def __get_input(self, path):
        image = tf.keras.preprocessing.image.load_img(path)
        image_arr = tf.keras.preprocessing.image.img_to_array(image)
        image_arr = tf.keras.preprocessing.image.smart_resize(image_arr, img_size)
        return image_arr

    def __getitem__(self, index):
        start = index * self.batch_size
        end = (index + 1) * self.batch_size
        batches = self.df[start:end]
        names = batches['Name']
        images = [self.__get_input(img) for img in names]  # images
        hc_features = [self.features[i] for i in batches['Loc']]  # données
        y = tf.keras.utils.to_categorical(batches['Label'], num_classes=2)
        images = np.asarray(images)
        hc_features = np.asarray(hc_features)
        # return [tf.Tensor(images, shape=(n)), tf.Tensor(hc_features)], y
        return [images, hc_features], y
        # return images, y

    def __len__(self):
        return self.n // self.batch_size
