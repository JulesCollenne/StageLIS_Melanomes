import numpy as np

import cfg
from dataset import get_generators
from models import build_model
from visu import plot_and_save
from dataset import get_testset
from visu import save_CM

training_generator, test_generator = get_generators(cfg.PATH)

model = build_model(cfg.NUM_CLASSES)

hist = model.fit(training_generator, use_multiprocessing=True,
                 workers=cfg.n_cpu, epochs=cfg.epochs, class_weight=cfg.class_weight)

plot_and_save(hist)

# print(model.evaluate(test_generator, verbose=2))
X_test, y_test = get_testset()
# X_test += mel_augmentation(X_test, y_test)
# X_test = np.asarray(X_test)

loss, acc = model.evaluate(X_test, y_test, verbose=2)
print("Test accuracy:")
print(acc)

y_pred = model.predict(X_test)
save_CM(np.argmax(y_pred, axis=1), np.argmax(y_test, axis=1))

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
