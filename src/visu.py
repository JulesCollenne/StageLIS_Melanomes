import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from features import get_points, get_axes, make_quadrants

quad_rgb = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]

# Affiche la courbe de l'accuracy
def plot_and_save(hist):
    plt.plot(hist.history['accuracy'])
    plt.title("Model accuracy")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.savefig("Courbes")


# Affiche la matrice de confusion
def save_CM(y_pred, y_true, labels=None):
    if labels is None:
        labels = ['NEV', 'MEL']
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot()
    plt.savefig("CM")


# Affiche les points colorés des quadrants
def show_mask_quad(points, quadrants):
    plt.scatter(points[:, 0], points[:, 1], c=quadrants, s=0.05)


# Renvoie une image.
# Représente les points colorés des quadrants avec du noir autour
def get_img_quadrant(mask, points, quad_rgb, quadrants):
    img_points = np.zeros(mask.shape, dtype=np.uint8)
    for num, point in enumerate(points):
        i = point[0]
        j = point[1]
        img_points[mask.shape[0] - j][i] = np.asarray(quad_rgb[quadrants[num]])
    return img_points


def mask2rgb(mask):
    return cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)


def quad_on_img(image, img_points, alpha):
    res = np.zeros((image.shape), dtype=np.uint8)
    for i, row in enumerate(img_points):
        for j, pixel in enumerate(row):
            if pixel.any() == 0:
                res[i][j] = image[i][j]
            else:
                res[i][j] = image[i][j] * alpha + img_points[i][j] * (1 - alpha)
    return res


# image : l'image de la lésion en couleur
# mask : le masque en noir et blanc de cette lésion
def example_draw_quadrants_on_lesion(image, mask):
    points = np.asarray(get_points(mask))
    axe1, axe2 = get_axes(points)
    quadrants = make_quadrants(points, axe1, axe2)
    img_points = get_img_quadrant(mask, points, quad_rgb, quadrants)
    final_img = quad_on_img(image, img_points, 0.8)
    cv2.imwrite('Quadrants.jpg', final_img)
