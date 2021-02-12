import math
import scipy.fft
import numpy as np
from skimage.feature import greycomatrix, greycoprops
import cv2


# Shape features

def get_major_axe(img):
    return 0


def get_minor_axe(img):
    return 0


def get_perimeter(img):
    return 0


def get_area(img):
    return 0


# Border irregularity index
def get_BII(img):
    a = get_major_axe(img)
    b = get_minor_axe(img)
    P = get_perimeter(img)
    A = get_area(img)
    return (a * b * P * P) / (2 * math.pi * (a * a + b * b) * A)


def get_GLCM(img):
    return greycomatrix(img, distances=[5], angles=[0], levels=256,
                        symmetric=True, normed=True)


######

# Calcul optimal du centroid d'une liste de point
def centeroidnp(arr):
    length = arr.shape[0]
    sum_x = np.sum(arr[:, 0])
    sum_y = np.sum(arr[:, 1])
    return sum_x / length, sum_y / length


def get_centroid(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    wi = width / 2
    he = height / 2
    ret, thres = cv2.threshold(gray, 95, 255, 0)
    M = cv2.moments(thres)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return cX, cY


def center_img(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    wi = width / 2
    he = height / 2
    ret, thres = cv2.threshold(gray, 95, 255, 0)
    M = cv2.moments(thres)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    cv2.circle(gray, (round(cX), round(cY)), 2, (0, 255, 0), -1)
    offsetX = wi - cX
    offsetY = he - cY
    T = np.float32([[1, 0, offsetX], [0, 1, offsetY]])
    centered_img = cv2.warpAffine(gray, T, (width, height))
    return centered_img


######


# Nos features

# Renvoie la tranformation de Fourier rapide en 2 dimensions
# Attention, l'image résultante contient des nombres complexes.
# Pour éviter que cela pose soucis, ajouter np.abs(...)
# Aussi, faire np.fft.fftshift(...) avant le abs pour plus de clarté
# (Cela met la fréquence zéro au centre)
def get_fourier_transform(img):
    return scipy.fft.fft2(img)


def color_symmetry(img):
    pass


# Feature innovante
# Renvoie la différence entre l'image et sa copie miroir (des deux axes)
def difference_image(img):
    l = img.shape[0]
    L = img.shape[1]
    n_channels = img.shape[2]
    diff = np.zeros((l, L, n_channels))
    for i in range(l):
        for j in range(L):
            for c in range(n_channels):
                diff[i][j][c] = np.abs(img[i][j][c] - img[l - i - 1][L - j - 1][c])

    return diff
