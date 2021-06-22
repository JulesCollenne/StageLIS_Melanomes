from CNN.models import get_efficientnet
from metrics.metrics_utils import visu_results
from src.handcrafted_models.models import get_HCANN
from src.loading.load_features import *


def load_test_set_multimodal():
    base = '/content/ISIC_2019/'
    folder = 'TEST/'
    path = base + 'NON_SEGMENTEES/ ' + folder
    X_testCNN = []
    for lesion_type in ('MEL', 'NEV'):
        current_path = path + lesion_type + '/'
        files = sorted([f for f in listdir(current_path) if isfile(join(current_path, f))])
        for file in files:
            img = cv2.resize(cv2.imread(current_path + file), img_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            X_testCNN.append(img)
    return np.asarray(X_testCNN)


def late_fusion(predictions, weights=None):
    return np.mean([predictions[i] * weights[i] for i in range(len(predictions))], axis=0)


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

# ------ Late fusion ------

predictions = [y_predANN, y_predCNN]
weights = (1, 1)
y_pred = late_fusion(predictions, weights=weights)

visu_results(y_testANN, y_pred, confidence=y_pred)
