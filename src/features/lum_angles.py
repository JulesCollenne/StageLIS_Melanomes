import cv2
import numpy as np

from distareas import get_lum_centroids
from features import scalar_feature_diff
from features_utils import get_center


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
