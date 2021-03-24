import math

import matplotlib.pyplot as plt
import numpy as np
from numpy import cos, sin, sqrt
from shapely.geometry import LineString


# Renvoie la liste de rayons
# Je fais un rayon en plus, et je jete le plus grand qui ne contient qu'un pixel
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
        for i in range(1, n_lines + 1):
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
