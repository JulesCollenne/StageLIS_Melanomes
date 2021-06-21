import cv2
import cv2
import numpy as np
from numpy import cos, sin
from sklearn.decomposition import PCA


# In : Grayscale image
# Out : Number of black pixels
from cfg import quad_rgb


def get_area(img):
    return len([pixel for row in img for pixel in row if pixel == 0])


# In : Grayscale image
# Out : Numpy array of points representing white pixels of the image
def get_points(img):
    return np.asarray([(j, i) for i, row in enumerate(img) for j, pixel in enumerate(row) if pixel != 0])


# In : Array of points
# Out : Coordinates of the centroid
def get_center(points):
    length = points.shape[0]
    sum_x = np.sum(points[:, 0])
    sum_y = np.sum(points[:, 1])
    return sum_x / length, sum_y / length


# In : Grayscale image (mask of the lesion)
# Out : Angle of rotation minimizing the area difference between the two halves of image
def get_axis_area(mask, center):
    aires = {}
    cX, cY = center
    for angle in range(0, 180, 10):
        res = rotate_img(mask, -angle, (cX, cY))
        taille = int(res.shape[0] / 2)
        aires[angle] = (res[:taille][:] != res[taille:][:]).sum()

    areas = np.argmin([aires[i] for i in aires.keys()])
    best_angle = areas * 10
    return best_angle


# In : Point p, line d
# Out : True if determinant is negative, False otherwise
def relative_side_of_point(p, d):
    return (d[0][0] - p[0]) * (d[1][1] - p[1]) - (d[0][1] - p[1]) * (d[1][0] - p[0]) < 0


def compute_PCA(points):
    center = get_center(points)
    cX, cY = center
    pca = PCA(n_components=2).fit(points)
    axes = []
    for i, (comp, var) in enumerate(zip(pca.components_, pca.explained_variance_ratio_)):
        comp = comp * var * 400
        axes.append([[cX, cY], [cX + comp[0], cY + comp[1]]])
    return np.asarray(axes)


def compute_Minim_area(points, mask):
    center = get_center(points)
    cX, cY = center
    best_angle = get_axis_area(mask, center)

    enX = int(400 * cos(-best_angle))
    enY = int(400 * sin(-best_angle))

    axe1 = ((cX - enX, cY - enY), (cX + enX, cY + enY))
    axe2 = ((cX + enY, cY - enX), (cX - enY, cY + enX))
    return np.asarray(axe1), np.asarray(axe2)


# In : List of points, method used to compute the symmetry axes
# Out : Symmetry axes
# method possible values : PCA, Minim
# PCA : Compute axes using PCA
# Minim : Compute axes using area difference minimization
def get_axes(points, method='PCA', mask=None):
    if method == 'PCA':
        return compute_PCA(points)
    elif method == 'Minim':
        if mask is None:
            raise ValueError("ERROR : Minim method needs a mask !")
        return compute_Minim_area(points, mask)
    else:
        raise ValueError("ERROR : Unknown method "+method)


# In : Mask of the lesion, Points, axe1, axe2
# Out : Quadrants of the image (mask.shape)
# Quadrants has 4 possible values :
# -1 : Not in the lesion, 0,1,2,3 : Quad number of the (i,j) pixel
def make_quadrants(mask, points, axe1, axe2):
    quadrants = np.ones((mask.shape[0], mask.shape[1]), dtype=np.uint8) * -1
    for p in points:
        i = p[1]
        j = p[0]
        if relative_side_of_point(p, axe1):
            if relative_side_of_point(p, axe2):
                quadrants[i][j] = 0
            else:
                quadrants[i][j] = 1
        else:
            if relative_side_of_point(p, axe2):
                quadrants[i][j] = 2
            else:
                quadrants[i][j] = 3
    return quadrants


# In : Image, quadrants, alpha
# Out : Resulting image of the fusion between alpha*img + (1-alpha)*quadrants
# alpha is the percentage of the original image
def quad_on_img(img, quadrants, alpha):
    img_points = np.zeros(img.shape, dtype=np.uint8)
    for i, row in enumerate(img):
        for j, pixel in enumerate(row):
            img_points[i][j] = np.asarray(quad_rgb[quadrants[i][j]])

    res = np.zeros((img.shape), dtype=np.uint8)
    for i, row in enumerate(img_points):
        for j, pixel in enumerate(row):
            if pixel.any() == -1:
                res[i][j] = img[i][j]
            else:
                res[i][j] = img[i][j] * alpha + img_points[i][j] * (1 - alpha)
    return res


# In : 2 points (x,y)
# Out : Euclidian distance between the two points
def distance(p1, p2):
    return np.sqrt(pow(p1[0] - p2[0], 2) + pow(p1[1] - p2[1], 2))


# In : Image, angle in degree, scale
# Out : Rotated image
def rotate_img(image, angle, center, scale=1.):
    (h, w) = image.shape[:2]
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated
