import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import auc
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import GridSearchCV, cross_val_score

from DataLoader import DataLoader


def visu_results(y_test, predictions, normalize=None, confidence=None):
    y_test = np.asarray(y_test)
    if len(y_test.shape) == 1:
        y_testOH = tensorflow.keras.utils.to_categorical(y_test)
    else:
        y_testOH = y_test
    if len(predictions.shape) == 1:
        predictions = tensorflow.keras.utils.to_categorical(predictions)
    cm = confusion_matrix(np.argmax(y_testOH, axis=1), np.argmax(predictions, axis=1), normalize=normalize)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=('Melanome', 'Naevus'))
    disp.plot()
    print(classification_report(np.argmax(y_testOH, axis=1),
                                np.argmax(predictions, axis=1),
                                target_names=('Mélanome', 'Naevus')))
    if confidence is None:
        print('ROC score on test set : ', roc_auc_score(y_testOH, predictions))
        fpr, tpr, thresh = roc_curve(np.asarray(y_testOH)[:, 0], np.asarray(predictions)[:, 0])
    else:
        print('ROC score on test set : ', roc_auc_score(y_test, confidence))
        fpr, tpr, thresh = roc_curve(np.argmax(np.asarray(y_test), axis=1), np.asarray(confidence)[:, 1])
    auc_keras = auc(fpr, tpr)
    plt.savefig("../../out/cm.png")
    plt.clf()
    plt.cla()
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label='Area = {:.3f}'.format(auc_keras))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='lower right')
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.savefig("../../out/roc.png")


def plot_hist(hist, metrics='accuracy'):
    plt.plot(hist.history[metrics], label=metrics)
    plt.plot(hist.history['val_' + metrics], label='val_' + metrics)
    plt.xlabel('Epoch')
    plt.ylabel(metrics)
    plt.legend(loc='lower right')


def run_and_test(featuresNames, parameters, clf, scoring='roc_auc', hasConfidence=False, centering=False):
    print('Loading features...')
    loader = DataLoader(featuresNames)
    X_train, y_train, X_test, y_test = loader.load()
    print('GridSeach...')
    gs = GridSearchCV(clf, parameters, scoring=scoring, n_jobs=-1, verbose=10).fit(X_train, y_train)

    print(gs.best_score_)
    print(gs.best_estimator_)
    clf = gs.best_estimator_

    print('Best estimator...')
    clf.fit(X_train, y_train)

    predictions = clf.predict(X_test)
    if hasConfidence:
        confidence = clf.decision_function(X_test)
        visu_results(y_test, predictions, confidence=confidence)
    else:
        visu_results(y_test, predictions)

    print('Cross_val_score (auc):', np.mean(cross_val_score(clf, X_train, y_train, scoring=scoring)))


def CNN_predict(model, dataset):
    y_true = []
    y_pred = []
    for X, y in dataset:
        y_true += y.numpy().tolist()
        y_pred += model(X).numpy().tolist()
    return y_true, y_pred
