import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_and_save(hist):
    plt.plot(hist.history['accuracy'])
    plt.title("Model accuracy")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.savefig("Courbes")


def save_CM(y_pred, y_true, labels=None):
    if labels is None:
        labels = ['NEV', 'MEL']
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot()
    plt.savefig("CM")
