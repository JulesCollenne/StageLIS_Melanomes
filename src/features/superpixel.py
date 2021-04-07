import math

import matplotlib.pyplot as plt
import numpy as np
import skimage
from numpy import cos, sin, sqrt
from shapely.geometry import LineString, Point
import cv2
from skimage.segmentation import slic

from features_utils import get_points, get_center, get_axes, make_quadrants


# Renvoie la liste de rayons
# Je fais un rayon en plus, et je jette le plus grand qui ne contient qu'un pixel
def get_radius_list(center, img, n_circles):
    maxi = 0
    for i, row in enumerate(img):
        for j, quad in enumerate(row):
            if quad != -1:
                dist = distance((i, j), (center[1], center[0]))
                if dist > maxi:
                    maxi = dist
    return [maxi / (n_circles + 1) * i for i in range(1, n_circles + 1)]


# Les 4 axes de base, créés à partir des deux vecteurs
def get_axes_list(axes):
    axes = np.asarray(axes)
    return axes[0], -axes[1], -axes[0], axes[1]


def rotate_point(point, degrees, center):
    degrees = math.radians(degrees)
    s = sin(degrees)
    c = cos(degrees)

    pt = point.copy()

    pt[0] -= center[1]
    pt[1] -= center[0]
    newX = pt[0] * c - pt[1] * s
    newY = pt[0] * s + pt[1] * c
    pt[0] = newX + center[1]
    pt[1] = newY + center[0]
    return pt


def rotate_axe(axe, degrees, center):
    return [rotate_point(point, degrees, center) for point in axe]
    # return rotate_point(axe[0], degrees, center), rotate_point(axe[1], -degrees, center)


def toLongLine(line, radius, center):
    newLine = []
    a = line[0]
    b = line[1]
    lenLine = sqrt(pow(a[0] - b[0], 2) + pow(a[1] - b[1], 2))
    newLine.append(a - (b - a) / lenLine * radius)
    newLine.append((center[0], center[1]) + (b - a) / lenLine * radius)
    return newLine


def toLongLines(axes, radius, center):
    return [toLongLine(axe, radius, center) for axe in axes]


# Ici, il faut calculer les axes qui passent par le centre de la lésion
# Selon leur nombre par quadrant, il faut calculer l'angle
# Qui les sépare, puis leur vecteur directeur en fonction du centre
# De la lésion
def get_line_list(center, axes, n_lines):
    # plot_lines(axes)
    line_list = []
    degreeToMove = 90. / (n_lines + 1)
    for axe in axes:
        for i in range(0, n_lines + 1):
            # plot_line(rotate_axe(axe, degreeToMove*i, center))
            value = rotate_axe(axe, degreeToMove * i, center)
            line_list.append(np.asarray(value))
    return np.asarray(line_list)


def distance(p1, p2):
    return np.sqrt(pow(p1[0] - p2[0], 2) + pow(p1[1] - p2[1], 2))


def plot_line(line):
    ligne = LineString(line)
    plt.plot(np.asarray(ligne.coords)[:, 0], np.asarray(ligne.coords)[:, 1])


def plot_lines(lines):
    for line in lines:
        plot_line(line)


def f_superpixel_circles_new_debug(img, center, axes, full_quads, numSegments=200, n_circles=2, n_lines=1):
    toutlespoints = []
    segments = slic(img, n_segments=numSegments, sigma=5)
    img2 = skimage.color.label2rgb(segments, img, kind='avg', bg_label=-1)
    feature = []
    cX, cY = center
    radius_list = get_radius_list((cY, cX), full_quads, n_circles)
    newAxes = toLongLines(axes, max(radius_list), (cX, cY))
    plot_lines(newAxes)
    line_list = get_line_list((cY, cX), newAxes, n_lines)
    for radius in radius_list:
        for line in line_list:
            p = Point(cX, cY)
            c = p.buffer(radius).boundary
            l = LineString(line)
            i = c.intersection(l)
            plt.scatter(np.asarray(c.coords)[:, 0], np.asarray(c.coords)[:, 1], s=1)
            plot_line(line)
            p1 = i.geoms[0].coords[0]
            p2 = i.geoms[1].coords[0]
            p1 = [int(i) for i in p1]
            p2 = [int(i) for i in p2]
            toutlespoints.append(p1)
            toutlespoints.append(p2)
            plt.scatter((p1[0], p2[0]), (p1[1], p2[1]))
            if (np.asarray(p1) >= 300).any() or (np.asarray(p2) >= 300).any():
                feature.append(np.asarray((0, 0, 0), dtype=np.uint8))
            else:
                a = np.asarray(img2[p1[1]][p1[0]], dtype=np.int16)
                b = np.asarray(img2[p2[1]][p2[0]], dtype=np.int16)
                feature.append(np.abs(a - b))
    print(toutlespoints)
    return np.asarray(feature).flatten()


if __name__ == "__main__":
    IMG_SIZE = (300, 300)
    win_name = 'Coucou toi, je suis un melanome'
    path = '../../images/melanoma.jpg'
    mask_path = '../../images/melanomask.jpg'
    img = cv2.resize(cv2.imread(path), (300, 300))
    mask = cv2.resize(cv2.imread(mask_path, 0), (300, 300))
    features = []
    points = np.asarray(get_points(mask))
    cX, cY = get_center(points)
    axe1, axe2 = get_axes(points)
    quadrants = make_quadrants(mask, points, axe1, axe2)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    feature = f_superpixel_circles_new_debug(img, (cX, cY), (axe1, axe2), quadrants)
    plt.gca().set_aspect('equal', adjustable='box')
    print(feature)
    segments = slic(img, n_segments=200, sigma=5)
    img2 = np.float32(skimage.color.label2rgb(segments, img, kind='avg', bg_label=-1))
    img2 = np.uint8(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    plt.imshow(img2)
    outpath = '../../out/'
    plt.savefig(outpath+'test.jpg')
