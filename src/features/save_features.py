import numpy as np

from beta import f_beta_quad
from geometry import get_points, get_center, get_axes, make_quadrants
from load_features import load_all_data

'''
    Code permettant la sauvegarde des features sous format txt.
    Généralement, la sauvegarde consiste en un fichier représentant
    une seule feature, par exemple X_train_beta_quad.txt.
    Les images sont ordonnées par leur nom à chaque fois
    donc la feature 0 représentera toujours la même image et ainsi
    de suite.
'''


def get_features(img, mask):
    points = np.asarray(get_points(mask))
    center = get_center(points)
    axe1, axe2 = get_axes(points, method='PCA', mask=mask)
    quadrants = make_quadrants(mask, points, axe1, axe2)

    # Pour calculer une autre feature, il suffit de changer la fonction
    # ci dessous
    features = f_beta_quad(img, quadrants)

    return features


# Emplacement de la sauvegarde
path = '/content/drive/MyDrive/Stage_LIS/Features/'
name = 'beta_quad_RGB'

X, y = load_all_data('TRAIN/', verbose=True, get_features=get_features)
np.savetxt(path + 'X_train_ ' + name + '.txt', X)

X, y = load_all_data('TEST/', verbose=True, get_features=get_features)
np.savetxt(path + 'X_test_ ' + name + '.txt', X)
