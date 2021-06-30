from os import listdir
from os.path import isfile, join
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras import Model

import cfg
from CNN.models import get_efficientnet
from handcrafted_models.models import get_HCANN

names = ('b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7')
fine_names = ('b0', 'b1', 'b2', 'b3', 'b4')
basic_names = ('b5', 'b6', 'b7')

hcann = get_HCANN()
hcann.load_weights(cfg.WEIGHTS_PATH + 'hcann.h5')

hcann_features = Model(hcann.input, hcann.layers[-2].output)

les_cnn = []

for name in fine_names:
    model = get_efficientnet(name, data_aug=False)
    model.load_weights(cfg.WEIGHTS_PATH + name + '_fine.h5')
    les_cnn.append(Model(model.input, model.layers[-2].output))

for name in basic_names:
    model = get_efficientnet(name, data_aug=False)
    model.load_weights(cfg.WEIGHTS_PATH + name + '.h5')
    les_cnn.append(Model(model.input, model.layers[-2].output))

CNN_features_path = "./"

chosen_CNNs = (0, 1, 2, 3, 4, 5, 6, 7)  # 0=EffNetB0, 1=EffNetB1 etc...

for group in ('TRAIN/', 'TEST/'):
    mypath = cfg.DATASET_PATH+group
    print(group)
    for model_num in chosen_CNNs:
        print("    Model " + str(model_num) + " en cours...")
        X_trainCNN = []
        for folder in ('MEL/', 'NEV/'):
            print("        " + folder)
            files = sorted([f for f in listdir(mypath + folder) if isfile(join(mypath + folder, f))])
            for i in range(len(files)):
                img = np.expand_dims(plt.imread(mypath + folder + files[i]), axis=0)
                X_trainCNN.append(les_cnn[model_num](img).numpy().squeeze())
        if group == 'TRAIN/':
            np.savetxt(CNN_features_path + "b" + str(model_num) + "_train_features.csv", X_trainCNN)
        else:
            np.savetxt(CNN_features_path + "b" + str(model_num) + "_test_features.csv", X_trainCNN)
