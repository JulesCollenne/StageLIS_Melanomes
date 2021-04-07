import numpy as np


# In : Image, quadrants, number of bins
# Out : Color histogram by quadrant, with n_bins
# Results are divided by the total number of pixel in the quadrant
from features import feature_diff


def get_histos(img, quadrants, n_bins=256):
    diviseur = 256 / n_bins
    histo = np.zeros((4, 3, n_bins))
    n_pixels = np.zeros(4)
    for i in range(quadrants.shape[0]):
        for j in range(quadrants.shape[1]):
            quad_num = int(quadrants[i][j])
            if quad_num != -1:
                for channel in range(3):
                    histo[quad_num][channel][int(img[i][j][channel] / diviseur)] += 1
                    n_pixels[quad_num] += 1
    return np.asarray([histo[i] / n_pixels[i] for i in range(4)])


# Feature permettant de calculer les différences entre les histogrammes de couleurs
# Prend en entrée l'image, les quadrants de l'image, ainsi que le nombre de bins
# Par défaut, le nombre de bin est de 10
# Les histogrammes sont sur les canaux RGB puis flatten.
# On a une liste de 4 histogrammes, sur lesquels on compare les valeurs entre chaque quadrants
def f_histo(img, quadrants, n_bins=10):
    histos = get_histos(img, quadrants, n_bins=n_bins)
    histos = histos.reshape((4, -1))
    return feature_diff(histos)
