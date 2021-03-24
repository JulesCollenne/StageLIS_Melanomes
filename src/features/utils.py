import math

import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy import round
from numpy.linalg import det
from sklearn.decomposition import PCA


def get_area(img):
    return len([pixel for row in img for pixel in row if pixel == 0])


def get_points(img):
    return np.asarray([(j, i) for i, row in enumerate(img) for j, pixel in enumerate(row) if pixel != 0])


def get_center(points):
    length = points.shape[0]
    sum_x = np.sum(points[:, 0])
    sum_y = np.sum(points[:, 1])
    return sum_x / length, sum_y / length


def get_axis_area(mask):
    points = np.asarray(get_points(mask))
    center = get_center(points)
    rows, cols = mask.shape
    M = np.float32([[1, 0, mask.shape[1] / 2 - center[0]], [0, 1, mask.shape[0] / 2 - center[1]]])
    mask = cv2.warpAffine(mask, M, (cols, rows))
    cY = int(center[0])
    cX = int(center[1])

    aires = {}
    for angle in range(0, 360, 10):
        res = rotate_img(mask, -angle, (cX, cY))
        taille = int(res.shape[0] / 2)
        aires[angle] = (res[:taille][:] != res[taille:][:]).sum()

    return np.argmin([aires[i] for i in aires.keys()]) * 10


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


def quad_on_img(img, img_points, alpha):
    res = np.zeros(img.shape, dtype=np.uint8)
    for i, row in enumerate(img_points):
        for j, pixel in enumerate(row):
            if pixel.any() == 0:
                res[i][j] = img[i][j]
            else:
                res[i][j] = img[i][j] * alpha + img_points[i][j] * (1 - alpha)
    return res


def points2img(points, img, quadrants):
    img_points = np.zeros(img.shape, dtype=np.uint8)
    for num, point in enumerate(points):
        i = point[0]
        j = point[1]
        img_points[j][i] = np.asarray(quad_rgb[quadrants[num]])
    return img_points


def plot_points(x, y, alpha=1, c=None, s=1):
    plt.scatter(x, y, alpha=alpha, c=c, s=s)
    ax = plt.gca()
    ax.set_ylim(ax.get_ylim()[::-1])
    plt.gca().set_aspect('equal', adjustable='box')


def center_img(gray):
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


def crop_lesion(mask, img):
    points = cv2.findNonZero(mask)
    x, y, w, h = cv2.boundingRect(points)
    return img[y:y + h, x:x + w]


def get_full_quads(quads, points, img):
    full_quads = np.ones((img.shape[0], img.shape[1])) * -1
    for i, (x, y) in enumerate(points):
        full_quads[y][x] = quads[i]
    return full_quads


def draw_mean_colors(full_quads, img, moyennes):
    result = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if full_quads[i][j] != -1:
                result[i][j] = img[i][j] * 0 + moyennes[int(full_quads[i][j])] * 1
            else:
                result[i][j] = img[i][j]
    return result


def get_angle(vector_1, vector_2):
    unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
    unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    return math.degrees(np.arccos(dot_product))


def rotate_img(image, angle, center, scale=1.):
    (h, w) = image.shape[:2]
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated


def cut_image(img):
    results = np.zeros((4, int(img.shape[0] / 2), int(img.shape[1] / 2), img.shape[2]))
    results[0] = img[:int(img.shape[0] / 2), int(img.shape[1] / 2):]
    results[1] = img[int(img.shape[0] / 2):, int(img.shape[1] / 2):]
    results[2] = img[:int(img.shape[0] / 2), :int(img.shape[1] / 2)]
    results[3] = img[int(img.shape[0] / 2):, :int(img.shape[1] / 2)]
    return results


def get_area_diff(mask):
    somme = 0
    for i in range(int(mask.shape[0] / 2)):
        for j in range(mask.shape[1]):
            somme += int(mask[i][j] != mask[mask.shape[0] - i - 1][j])
    return somme

def get_cropped(img, mask, points, axe1, axe2):
    center = get_center(points)
    vector = (axe1[1][0] - axe1[0][0], axe1[1][1] - axe1[0][1])
    vector2 = (axe2[1][0] - axe2[0][0], axe2[1][1] - axe2[0][1])
    angle = get_angle((1, 0), vector)
    img_rotated = rotate_img(img, -angle, center)
    mask_rotated = rotate_img(mask, -angle, center)
    cropped = crop_lesion(mask_rotated, img_rotated)
    cropped = cv2.resize(cropped, img_shape)
    return cropped

quad_rgb = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
