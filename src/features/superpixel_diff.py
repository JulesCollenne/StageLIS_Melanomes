import numpy as np
from skimage import color
from skimage.segmentation import slic


def f_superpixel_diff(images, numSegments=50):
    superpixelise = []
    for image in images:
        segments = slic(image, n_segments=numSegments, sigma=5, compactness=50)
        out1 = color.label2rgb(segments, image, kind='avg', bg_label=-1)
        superpixelise.append(out1)
    unique_colors = sorted([np.unique(np.reshape(quad, (-1, 3)), axis=0) for quad in superpixelise],
                           key=lambda rgb: np.mean(rgb))
    for i, quad in enumerate(unique_colors):
        while len(unique_colors[i]) < numSegments:
            unique_colors[i] = np.concatenate((unique_colors[i], np.asarray((unique_colors[i][-1]).reshape((1, 3)))))
        if len(unique_colors[i]) > 50:
            unique_colors[i] = unique_colors[i][:50]
    unique_colors = np.asarray(unique_colors).reshape((4, -1))
    # return feature_diff(unique_colors)
    diff = np.abs(unique_colors[0] - unique_colors[3], unique_colors[1] - unique_colors[2])
    return list(diff.flatten())
