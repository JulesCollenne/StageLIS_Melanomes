import numpy as np
import skimage

from scipy.stats import skew, kurtosis


# Basic features
# Mean, std, kurtosis, skewness and shannon entropy
from feature_difference import feature_diff


def f_beta(img):
    feature = [np.mean(img, axis=(0, 1)),
               np.std(img, axis=(0, 1)),
               kurtosis(kurtosis(img, axis=0), axis=0),
               skew(skew(img, axis=0), axis=0),
               skimage.measure.shannon_entropy(img)]
    return np.asarray(feature).flatten()


# Basic feature by quadrant
# Mean, std, kurtosis, skewness and shannon entropy
def f_beta_quad(img, quadrants):
    features = []
    for quad_num in range(4):
        tmp_img = [img[i][j] for i in range(img.shape[0]) for j in range(img.shape[1])
                   if quadrants[i][j] == quad_num]
        feature = [np.mean(tmp_img, axis=0),
                   np.std(tmp_img, axis=0),
                   kurtosis(tmp_img, axis=0),
                   skew(np.asarray(tmp_img), axis=0)]
        feature = list(np.asarray(feature).flatten())
        feature += skimage.measure.shannon_entropy(tmp_img)
        features.append(np.asarray(feature).flatten())
    return feature_diff(np.asarray(features))
