import math

import bcolors as bcolors
import cv2
import numpy as np
from numpy import cos, sin
from sklearn.decomposition import PCA
import cfg


# In : Grayscale image
# Out : Number of black pixels
def get_area(img):
    return len([pixel for row in img for pixel in row if pixel == 0])


# In : Grayscale image
# Out : Numpy array of points representing white pixels of the image
def get_points(img):
    return np.asarray([(j, i) for i, row in enumerate(img) for j, pixel in enumerate(row) if pixel != 0])


# In : Array of points
# Out : Coordinates of the mean coordinates
def get_center(points):
    length = points.shape[0]
    sum_x = np.sum(points[:, 0])
    sum_y = np.sum(points[:, 1])
    return sum_x / length, sum_y / length


# In : Grayscale image (mask of the lesion)
# Out : Angle of rotation minimizing the area difference between the two halves of image
def get_axis_area(mask, center):
    aires = {}
    cX, cY = center
    for angle in range(0, 180, 10):
        res = rotate_img(mask, -angle, (cX, cY))
        taille = int(res.shape[0] / 2)
        aires[angle] = (res[:taille][:] != res[taille:][:]).sum()

    areas = np.argmin([aires[i] for i in aires.keys()])
    best_angle = areas * 10
    return best_angle


# In : Point p, line d
# Out : True if determinant is negative, False otherwise
def relative_side_of_point(p, d):
    return (d[0][0] - p[0]) * (d[1][1] - p[1]) - (d[0][1] - p[1]) * (d[1][0] - p[0]) < 0


# In : List of points, method used to compute the symmetry axes
# Out : Symmetry axes
# method possible values : PCA, Minim
# PCA : Compute axes using PCA
# Minim : Compute axes using area difference minimization
def get_axes(points, method='PCA', mask=None):
    cX, cY = get_center(points)
    if method == 'PCA':
        pca = PCA(n_components=2).fit(points)
        axes = []
        for i, (comp, var) in enumerate(zip(pca.components_, pca.explained_variance_ratio_)):
            comp = comp * var * 400
            axes.append([[cX, cY], [cX + comp[0], cY + comp[1]]])
        return np.asarray(axes)
    elif method == 'Minim':
        if mask is None:
            print(
                bcolors.FAIL + 'Mask is None using Minim method !\nUse PCA if you dont want to use mask' + bcolors.ENDC)
        best_angle = get_axis_area(mask)

        enX = int(400 * cos(-best_angle))
        enY = int(400 * sin(-best_angle))

        axe1 = ((cX - enX, cY - enY), (cX + enX, cY + enY))
        axe2 = ((cX + enY, cY - enX), (cX - enY, cY + enX))
        return np.asarray(axe1), np.asarray(axe2)
    return None


# In : Mask of the lesion, Points, axe1, axe2
# Out : Quadrants of the image (mask.shape)
# Quadrants has 4 possible values : 
# -1 : Not in the lesion, 0,1,2,3 : Quad number of the (i,j) pixel
def make_quadrants(mask, points, axe1, axe2):
    quadrants = np.ones((mask.shape[0], mask.shape[1]), dtype=np.uint8) * -1
    for p in points:
        i = p[1]
        j = p[0]
        if relative_side_of_point(p, axe1):
            if relative_side_of_point(p, axe2):
                quadrants[i][j] = 0
            else:
                quadrants[i][j] = 1
        else:
            if relative_side_of_point(p, axe2):
                quadrants[i][j] = 2
            else:
                quadrants[i][j] = 3
    return quadrants


# In : Image, quadrants, alpha
# Out : Resulting image of the fusion between alpha*img + (1-alpha)*quadrants
# alpha is the percentage of the original image
def quad_on_img(img, quadrants, alpha):
    img_points = np.zeros(img.shape, dtype=np.uint8)
    for i, row in enumerate(img):
        for j, pixel in enumerate(row):
            img_points[i][j] = np.asarray(cfg.quad_rgb[quadrants[i][j]])

    res = np.zeros(img.shape, dtype=np.uint8)
    for i, row in enumerate(img_points):
        for j, pixel in enumerate(row):
            if pixel.any() == -1:
                res[i][j] = img[i][j]
            else:
                res[i][j] = img[i][j] * alpha + img_points[i][j] * (1 - alpha)
    return res


# In : 2 points (x,y)
# Out : Euclidian distance between the two points
def distance(p1, p2):
    return np.sqrt(pow(p1[0] - p2[0], 2) + pow(p1[1] - p2[1], 2))


# In : Image, angle in degree, scale
# Out : Rotated image
def rotate_img(image, angle, center, scale=1.):
    (h, w) = image.shape[:2]
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated


def get_angle(vector_1, vector_2):
    unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
    unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    return math.degrees(np.arccos(dot_product))


def crop_lesion(mask, img):
    points = cv2.findNonZero(mask)
    x, y, w, h = cv2.boundingRect(points)
    return img[y:y + h, x:x + w]


def get_cropped(img, mask, points, axe1, axe2):
    center = get_center(points)
    vector = (axe1[1][0] - axe1[0][0], axe1[1][1] - axe1[0][1])
    # vector2 = (axe2[1][0] - axe2[0][0], axe2[1][1] - axe2[0][1])
    angle = get_angle(np.asarray((1, 0)), np.asarray(vector))
    img_rotated = rotate_img(img, -angle, center)
    mask_rotated = rotate_img(mask, -angle, center)
    cropped = crop_lesion(mask_rotated, img_rotated)
    cropped = cv2.resize(cropped, cfg.img_shape)
    return cropped
