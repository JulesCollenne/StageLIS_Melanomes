import cv2
import numpy as np

from features import feature_diff


# In : Quadrants, image
# Out : Mean color for each quadrant
from utils import get_points, get_axes, make_quadrants


def get_mean_colors(quadrants, img):
    moyennes = []
    for quad_num in range(4):
        moyennes.append(np.mean([img[i][j] for i in range(img.shape[0]) for j in range(img.shape[1])
                                 if quadrants[i][j] == quad_num], axis=0))
    return moyennes


# Feature function
# In : Image, quadrants
# Out : Feature color mean
def f_color_mean(img, quadrants):
    moyennes = get_mean_colors(quadrants, img)
    return feature_diff(moyennes)


# In : Quadrants, image, moyennes des couleurs par quadrant
# Out : Image with mean color by quadrant
def draw_mean_colors(quadrants, img, moyennes):
    result = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if quadrants[i][j] != -1:
                result[i][j] = img[i][j] * 0 + moyennes[int(quadrants[i][j])] * 1
            else:
                result[i][j] = img[i][j]
    return result


if __name__ == "__main__":
    IMG_SIZE = (300,300)
    path = '/content/ISIC_2019/SEGMENTEES/HIGH_RESOL_FILLED/TEST/MEL/ISIC_0033710.JPG'
    mask_path = '/content/ISIC_2019/SEGMENTEES/MASK/TEST/MEL/ISIC_0033710_Mask.jpg'
    img = cv2.resize(cv2.imread(path), IMG_SIZE)
    mask = cv2.resize(cv2.imread(mask_path, 0), IMG_SIZE)
    points = np.asarray(get_points(mask))
    axe1, axe2 = get_axes(points)
    quadrants = make_quadrants(mask, points, axe1, axe2)
    moyennes = get_mean_colors(quadrants, img)
    cv2.imshow(img)
    print("Everything passed")
