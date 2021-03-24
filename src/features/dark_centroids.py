import numpy as np


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
