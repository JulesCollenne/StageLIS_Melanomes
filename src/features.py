import math
import scipy.fft
import numpy as np
from skimage.feature import greycomatrix, greycoprops
import cv2
from sklearn.decomposition import PCA


# Centre, axes de symmétrie

def get_area(img):
    return len([pixel for row in img for pixel in row if pixel == 0])


def get_points(img):
    return [(j, img.shape[0] - i - 1) for i, row in enumerate(img) for j, pixel in enumerate(row) if pixel != 0]


def get_center(points):
    length = points.shape[0]
    sum_x = np.sum(points[:, 0])
    sum_y = np.sum(points[:, 1])
    return sum_x / length, sum_y / length


def relative_side_of_point(p, d):
    return (d[0][0] - p[0]) * (d[1][1] - p[1]) - (d[0][1] - p[1]) * (d[1][0] - p[0]) < 0


def get_axes(points):
    cX, cY = get_center(points)
    pca = PCA(n_components=2).fit(points)
    axes = []
    for i, (comp, var) in enumerate(zip(pca.components_, pca.explained_variance_ratio_)):
        comp = comp * var * 400
        axes.append([[cX, cY], [cX + comp[0], cY + comp[1]]])
    return np.asarray(axes)


def make_quadrants(points, axe1, axe2):
    quadrants = []
    for p in points:
        if relative_side_of_point(p, axe1):
            if relative_side_of_point(p, axe2):
                quadrants.append(0)
            else:
                quadrants.append(1)
        else:
            if relative_side_of_point(p, axe2):
                quadrants.append(2)
            else:
                quadrants.append(3)
    return np.asarray(quadrants)


def get_full_quads(quads, points, img):
    full_quads = np.ones((img.shape[0], img.shape[1])) * -1
    for i, x, y in enumerate(points):
        full_quads[x][y] = quads[i]
    return full_quads


def get_quad_mean_color(quads, points, img):
    full_quads = get_full_quads(quads, points, img)
    moyennes = []

    for quad_num in (0, 1, 2, 3):
        moyennes.append(
            np.mean([img[i][j] for i in range(len(img)) for j in range(len(img[i])) if full_quads[i][j] == quad_num]))

    return moyennes


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
# Axis=0 : Horizontal
# Axis=1 : Vertical
# Axis=2 : Les deux (diagonale)
def get_diff(image, axis=2):
    if axis not in (0, 1, 2):
        print("Axis parameter is either 0, 1 or 2.")
        print("Please use a correct value")
        exit(1)
    diff = np.zeros((image.shape[0], image.shape[1], image.shape[2]))
    if axis == 2:
        for i in range(len(image)):
            for j in range(len(image[i])):
                for c in range(len(image[i][j])):
                    diff[i][j][c] = np.abs(image[i][j][c] - image[image.shape[0] - i - 1][image.shape[1] - j - 1][c])
    elif axis == 0:
        for i in range(int(len(image) / 2)):
            for j in range(len(image[i])):
                for c in range(len(image[i][j])):
                    diff[i][j][c] = np.abs(image[i][j][c] - image[image.shape[0] - i - 1][j][c])
    elif axis == 1:
        for i in range(len(image)):
            for j in range(int(len(image[i]) / 2)):
                for c in range(len(image[i][j])):
                    diff[i][j][c] = np.abs(image[i][j][c] - image[i][image.shape[1] - j - 1][c])
    return diff


def crop_lesion(mask, img):
    points = cv2.findNonZero(mask)
    x, y, w, h = cv2.boundingRect(points)
    return img[y:y + h, x:x + w]


def get_diff_img(mask, img, axis=2):
    cropped = crop_lesion(mask, img)
    return get_diff(cropped, axis)


########################## Inutilisé

###########

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


def get_major_axe(img):
    return 0


def get_minor_axe(img):
    return 0


def get_perimeter(img):
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


# Feature extractor algorithms

def get_sift_features(img):
    sift = cv2.SIFT()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp = sift.detect(gray, None)
    img = cv2.drawKeypoints(gray, kp)
    # cv2.imwrite('sift_keypoints.jpg',img)
    return kp
