import skimage
import skimage.measure
from numpy.linalg import det
from scipy.stats import kurtosis, skew
from shapely.geometry import LineString
from shapely.geometry import Point
from skimage import color
from skimage.segmentation import slic
from distareas import *
from superpixel_color import get_radius_list, toLongLines, get_line_list
from features_utils import *


def f_superpixel_diff(images, num_segments=50):
    superpixelise = []
    for image in images:
        segments = slic(image, n_segments=num_segments, sigma=5, compactness=50)
        out1 = skimage.color.label2rgb(segments, image, kind='avg', bg_label=-1)
        superpixelise.append(out1)
    unique_colors = sorted([np.unique(np.reshape(quad, (-1, 3)), axis=0) for quad in superpixelise],
                           key=lambda rgb: np.mean(rgb))
    for i, quad in enumerate(unique_colors):
        while len(unique_colors[i]) < num_segments:
            unique_colors[i] = np.concatenate((unique_colors[i], np.asarray((unique_colors[i][-1]).reshape((1, 3)))))
        if len(unique_colors[i]) > 50:
            unique_colors[i] = unique_colors[i][:50]
    unique_colors = np.asarray(unique_colors).reshape((4, -1))
    # return feature_diff(unique_colors)
    diff = np.abs(unique_colors[0] - unique_colors[3], unique_colors[1] - unique_colors[2])
    return list(diff.flatten())


def f_entropy(img):
    feature = []
    images = cut_image(img)
    for image in images:
        feature += skimage.measure.shannon_entropy(image)
    return feature


def get_areas(thresh, full_quads):
    somme = np.zeros((4, 2))
    for i in range(thresh.shape[0]):
        for j in range(thresh.shape[1]):
            if int(full_quads[i][j]) != -1:
                if thresh[i][j].all() == 0: # Todo : Honnetement je crois qu'il faut mettre 1 ici (et non pas 0)
                    somme[int(full_quads[i][j])][0] += 1
                somme[int(full_quads[i][j])][1] += 1
    return somme


def distance(p1, p2):
    return np.sqrt(pow(p1[0] - p2[0], 2) + pow(p1[1] - p2[1], 2))


# Angles entre les centroids de teintes sombre
def f_lum_angles(image, full_quads, points):
    ret, thresh = cv2.threshold(image, 85, 255, cv2.THRESH_BINARY)
    center = get_center(points)
    centroids = []
    for n_quad in range(4):
        centroids.append(get_lum_centroids(n_quad, full_quads, thresh))

    angles = []

    for quad in centroids:
        ba = quad[0] - center
        bc = quad[1] - center
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angles.append(np.abs(np.arccos(round(cosine_angle, 2))))
    return scalar_feature_diff(angles)


def f_beta(img):
    feature = [
        np.mean(img, axis=(0, 1)),
        np.std(img, axis=(0, 1)),
        kurtosis(kurtosis(img, axis=0), axis=0),
        skew(skew(img, axis=0), axis=0),
        np.ones(3) * skimage.measure.shannon_entropy(img)]
    return np.asarray(feature).flatten()


def f_beta_quad(imgs):
    features = []
    for img in imgs:
        feature = [np.mean(img, axis=(0, 1)),
                   np.std(img, axis=(0, 1)),
                   kurtosis(kurtosis(img, axis=0), axis=0),
                   skew(skew(img, axis=0), axis=0),
                   np.ones(3) * skimage.measure.shannon_entropy(img)]
        features.append(np.asarray(feature).flatten())
    return feature_diff(np.asarray(features))


# Peut etre calculer la variance de couleur sur un même cercle ?
def f_superpixel_circles(img, center, full_quads, numSegments=50):
    segments = slic(img, n_segments=numSegments, sigma=5)
    img2 = color.label2rgb(segments, img, kind='avg', bg_label=-1)
    feature = []
    cX, cY = center
    radius = (10, 15, 20)
    for rad in radius:
        somme = np.zeros((4, 3))
        nb_pixels = np.zeros(4)
        for i, row in enumerate(img2):
            for j, pixel in enumerate(row):
                if int(distance((i, j), (cX, cY))) == rad:
                    quad_num = int(full_quads[i][j])
                    somme[quad_num] += pixel
                    nb_pixels[quad_num] += 1

        if 0 in nb_pixels:
            print('Warning : missing pixel values')
        somme = [somme[i] / nb_pixels[i] if 0 != nb_pixels[i]
                 else np.mean(img2, axis=(0, 1), dtype=int)
                 for i in range(somme.shape[0])]
        feature += feature_diff(somme)
    return feature


def f_superpixel_circles_new(img, center, axes, full_quads, numSegments=200, n_circles=2, n_lines=1):
    segments = slic(img, n_segments=numSegments, sigma=5)
    img2 = skimage.color.label2rgb(segments, img, kind='avg', bg_label=-1)
    feature = []
    cX, cY = center
    radius_list = get_radius_list((cY, cX), full_quads, n_circles)
    newAxes = toLongLines(axes, max(radius_list), (cX, cY))
    line_list = get_line_list((cY, cX), newAxes, n_lines)
    for radius in radius_list:
        for line in line_list:
            p = Point(cX, cY)
            c = p.buffer(radius).boundary
            ligne = LineString(line)
            i = c.intersection(ligne)
            p1 = i.geoms[0].coords[0]
            p2 = i.geoms[1].coords[0]
            p1 = [int(i) for i in p1]
            p2 = [int(i) for i in p2]
            if (np.asarray(p1) >= 300).any() or (np.asarray(p2) >= 300).any():
                feature.append(np.asarray((0, 0, 0), dtype=np.uint8))
            else:
                a = np.asarray(img2[p1[0]][p1[1]], dtype=np.int16)
                b = np.asarray(img2[p2[0]][p2[1]], dtype=np.int16)
                feature.append(np.abs(a - b))
    return np.asarray(feature).flatten()


def get_features(img, mask):
    features = []
    points = np.asarray(get_points(mask))
    axe1, axe2 = get_axes(points)
    quadrants = make_quadrants(points, axe1, axe2)
    full_quads = get_full_quads(quadrants, points, img)
    cropped = get_cropped(img, mask, points, axe1, axe2)
    #features += f_color_mean(img, full_quads)
    features += f_histo(img, full_quads)
    features += f_distareas(img, full_quads, points)
    features += list(f_entropy(cropped))
    return features
