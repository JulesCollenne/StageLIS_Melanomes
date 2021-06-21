import numpy as np

from features_utils import crop_lesion


# Axis=0 : Horizontal
# Axis=1 : Vertical
# Axis=2 : Les deux (diagonale)
def get_diff(image, axis=2):
    if axis not in (0, 1, 2):
        print("Axis parameter is either 0, 1 or 2.")
        print("Please use a correct value")
        exit(1)
    image = image.astype('int32')
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


def get_diff_img(mask, img, axis=2):
    cropped = crop_lesion(mask, img)
    return get_diff(cropped, axis)
