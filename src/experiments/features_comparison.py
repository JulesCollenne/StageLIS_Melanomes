from sklearn.svm import LinearSVC

from DataLoader import DataLoader
from metrics.metrics_utils import visu_results

featuresNames = ('beta')
loader = DataLoader(featuresNames)

X_train, y_train, X_test, y_test = loader.load()

clf = LinearSVC(C=0.1, penalty='l2', class_weight={0:4,1:1})
clf.fit(X_train, y_train)

predictions = clf.predict(X_test)
confidence = clf.decision_function(X_test)

visu_results(y_test, predictions, confidence=confidence)
