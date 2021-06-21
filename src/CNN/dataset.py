import tensorflow as tf

scal_size = 600
img_size = (scal_size, scal_size)
img_dim = (scal_size, scal_size, 3)
batch_size = 16


def get_datasets(folder='NON_SEGMENTEES'):
    train_folder = '/TRAIN'
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        '/content/ISIC_2019/ ' + folder + train_folder,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='categorical')

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        '/content/ISIC_2019/ ' + folder + train_folder,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='categorical')

    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        '/content/ISIC_2019/ ' + folder + '/TEST',
        image_size=img_size,
        batch_size=batch_size,
        label_mode='categorical',
        seed=123)
    return train_ds, val_ds, test_ds





