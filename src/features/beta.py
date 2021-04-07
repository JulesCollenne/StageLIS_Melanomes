import numpy as np
import skimage

from scipy.stats import skew, kurtosis


# Basic features
# Mean, std, kurtosis, skewness and shannon entropy
from features import feature_diff


def f_beta(img, mask):
    feature = []
    feature.append(np.mean(img, axis=(0, 1)))
    feature.append(np.std(img, axis=(0, 1)))
    feature.append(kurtosis(kurtosis(img, axis=0), axis=0))
    feature.append(skew(skew(img, axis=0), axis=0))
    feature.append(skimage.measure.shannon_entropy(img))
    return np.asarray(feature).flatten()


# Basic feature by quadrant
# Mean, std, kurtosis, skewness and shannon entropy
def f_beta_quad(imgs, mask):
    features = []
    for img in imgs:
        feature = []
        feature.append(np.mean(img, axis=(0, 1)))
        feature.append(np.std(img, axis=(0, 1)))
        feature.append(kurtosis(kurtosis(img, axis=0), axis=0))
        feature.append(skew(skew(img, axis=0), axis=0))
        feature.append(skimage.measure.shannon_entropy(img))
        features.append(np.asarray(feature).flatten())
    return feature_diff(np.asarray(features))
