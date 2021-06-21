import numpy as np


# In : Any feature by quadrant (list of list with shape (4,x))
# Out : Mean differences for each quadrant relative to the axes, and center
# Output shape : (3*x)
def feature_diff(feature):
    result = []
    feature = np.asarray(feature)

    somme = (np.abs(feature[0] - feature[3]) + np.abs(feature[1] - feature[2])) / 2
    result += list(somme)

    somme = (np.abs(feature[0] - feature[1]) + np.abs(feature[2] - feature[3]) + np.abs(
        feature[0] - feature[2]) + np.abs(feature[1] - feature[3])) / 4
    result += list(somme)

    # Moyenne des différences par rapport aux axes
    H = np.abs((feature[0] + feature[1]) / 2 - (feature[2] + feature[3]) / 2)
    V = np.abs((feature[0] + feature[2]) / 2 - (feature[2] + feature[1]) / 2)
    result += list((H + V) / 2)
    return result


# In : Any feature by quadrant except the feature is a scalar
# Out : Mean differences for each quadrant relative to the axes, and center
def scalar_feature_diff(feature):
    result = []
    feature = np.asarray(feature)

    somme = (np.abs(feature[0] - feature[3]) + np.abs(feature[1] - feature[2])) / 2
    result.append(somme)

    somme = (np.abs(feature[0] - feature[1]) + np.abs(feature[2] - feature[3]) + np.abs(
        feature[0] - feature[2]) + np.abs(feature[1] - feature[3])) / 4
    result.append(somme)

    # Moyenne des différences par rapport aux axes
    H = np.abs((feature[0] + feature[1]) / 2 - (feature[2] + feature[3]) / 2)
    V = np.abs((feature[0] + feature[2]) / 2 - (feature[2] + feature[1]) / 2)
    result.append((H + V) / 2)
    return result
