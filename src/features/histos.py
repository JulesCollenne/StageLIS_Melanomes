import cv2
import numpy as np
import matplotlib.pyplot as plt

from feature_difference import feature_diff
from geometry import get_axes
from geometry import get_points, make_quadrants, quad_on_img


# In : Image, quadrants, number of bins
# Out : Color histogram by quadrant, with n_bins
# Results are divided by the total number of pixel in the quadrant
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


if __name__ == "__main__":
    IMG_SIZE = (300, 300)
    path = '/content/ISIC_2019/SEGMENTEES/HIGH_RESOL_FILLED/TEST/MEL/ISIC_0000165.JPG'
    mask_path = '/content/ISIC_2019/SEGMENTEES/MASK/TEST/MEL/ISIC_0000165_Mask.jpg'
    img = cv2.resize(cv2.imread(path), IMG_SIZE)
    mask = cv2.resize(cv2.imread(mask_path, 0), IMG_SIZE)
    points = np.asarray(get_points(mask))
    axe1, axe2 = get_axes(points, method='Minim', mask=mask)
    quadrants = make_quadrants(mask, points, axe1, axe2)
    res = quad_on_img(img, quadrants, 0.8)
    cv2.imshow("Quadrants", res)
    histos = get_histos(img, quadrants, n_bins=10)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    fig, axs = plt.subplots(3)
    axs[0].bar(np.linspace(0, 256, 10), histos[0][0], width=10, color='blue')
    axs[1].bar(np.linspace(0, 256, 10), histos[0][1], width=10, color='green')
    axs[2].bar(np.linspace(0, 256, 10), histos[0][2], width=10, color='red')
