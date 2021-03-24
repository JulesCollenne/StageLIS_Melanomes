import keras
from keras.layers import Concatenate, Dense, Input
from keras.models import Model
import cfg

in1 = Input(shape=(cfg.IMG_SIZE, cfg.IMG_SIZE, 3))
cnn = keras.models.load_model('../out/models/cnn')

in2 = Input(shape=()) # Mettre la dim des features ici
nn = keras.models.load_model('../out/models/nn')

final_nn = Concatenate(axis=-1)([cnn, nn])
final_nn = Dense(2, activation='softmax')(final_nn)

X_train_img = []
X_train_feat = []

model = Model(inputs=[in1, in2], outputs=final_nn)

model.compile(loss='categorical_crossentropy',  # continu together
              optimizer='adam',
              metrics=['accuracy'])

model.fit([X_train_img, X_train_feat], Y_train,
          batch_size=32, nb_epoch=10, verbose=1)
