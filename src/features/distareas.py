import cv2
import numpy as np

from features import get_areas, scalar_feature_diff
from features_utils import get_center


def get_lum_centroids(num_quad, full_quads, thresh):
    clair = []
    sombre = []
    total = []
    for i, row in enumerate(thresh):
        for j, pixel in enumerate(row):
            if full_quads[i][j] == num_quad:
                total.append((j, i))
                if pixel.all() != 0:
                    clair.append((j, i))
                elif pixel.all() == 0:
                    sombre.append((j, i))
    # clairMean = np.mean(clair, axis=0, dtype=np.int32)
    totalMean = np.mean(total, axis=0, dtype=np.int32)
    if len(sombre) == 0:
        sombreMean = totalMean
    else:
        sombreMean = np.mean(sombre, axis=0, dtype=np.int32)
    return totalMean, sombreMean


# d1 : total d2 : sombre
# aire0 : sombre aire1 : total
def f_distareas(image, full_quads, points):
    ret, thresh = cv2.threshold(image, 85, 255, cv2.THRESH_BINARY)
    feature = []
    center = get_center(points)
    cX = center[0]
    cY = center[1]
    d1 = []
    d2 = []
    centroids = []
    for n_quad in range(4):
        centroids.append(get_lum_centroids(n_quad, full_quads, thresh))

    for quad in centroids:
        d1.append(np.linalg.norm(quad[0] - (cX, cY)))
        d2.append(np.linalg.norm(quad[1] - (cX, cY)))

    areas = get_areas(thresh, full_quads)

    feature = [(areas[i][0] * d2[i]) / (areas[i][1] * d1[i]) for i in range(4)]
    # for i in range(4):
    # feature += (areas[i][0] * d2[i]) / (areas[i][1] * d1[i])

    return scalar_feature_diff(feature)
