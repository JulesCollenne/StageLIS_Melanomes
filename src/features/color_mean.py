import numpy as np


def get_mean_colors(full_quads, img):
    moyennes = []
    for quad_num in range(4):
        moyennes.append(np.mean([img[i][j] for i in range(img.shape[0]) for j in range(img.shape[1])
                                 if full_quads[i][j] == quad_num], axis=0))
    return moyennes
