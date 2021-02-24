import numpy as np

import cfg
from dataset import get_set
from models import build_model
from visu import plot_and_save, writePredictions
from visu import save_CM

X_train, y_train, _ = get_set('TRAIN')
X_test, y_test, testImgNames = get_set('TEST')

model = build_model(cfg.NUM_CLASSES)

hist = model.fit(X_train, y_train, use_multiprocessing=True,
                 workers=cfg.n_cpu, epochs=cfg.epochs, class_weight=cfg.class_weight,
                 validation_split=0.1)

plot_and_save(hist)

loss, acc = model.evaluate(X_test, y_test, verbose=2)
print("Test accuracy:")
print(acc)

y_pred = model.predict(X_test)
save_CM(np.argmax(y_pred, axis=1), np.argmax(y_test, axis=1))

writePredictions(y_pred, y_test, testImgNames)

model.save("../out/models/subset")

'''
y_pred = model.predict(test_generator)
y_true = np.concatenate([np.argmax(test_generator[i][1], axis=1) for i in range(cfg.batch_size)])

print(np.argmax(y_pred, axis=1).shape)
print(y_true.shape)
print("Nombre de naevus prédit : ")
print(len([pred for pred in y_pred if pred == 0]))
print("Nombre total de prédiction : ")
print(len(y_pred))
'''

# save_CM(np.argmax(y_pred, axis=1), y_true)
