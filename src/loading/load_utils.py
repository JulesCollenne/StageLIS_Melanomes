import cv2


def change_space_color(img, mcolor):
    if mcolor == 'bgr':
        return img
    elif mcolor == 'hsv':
        return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    elif mcolor == 'cielab':
        return cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    elif mcolor == 'ycrcb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    else:
        raise Exception('Correct values for color_models are : bgr, hsv, cielab, ycrcb\nYou provided : ', mcolor)
