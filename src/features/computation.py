import os
from os import listdir
from os.path import isfile, join

import cv2
import cv2 as cv

import cfg
from features.geometry import *


def get_features(img, mask, color_models=('bgr')):
    features = []
    points = np.asarray(get_points(mask))
    center = get_center(points)
    axe1, axe2 = get_axes(points)
    quadrants = make_quadrants(mask, points, axe1, axe2)
    for mcolor in color_models:
        if mcolor == 'rgb':
            pass
        elif mcolor == 'hsv':
            img = cv.cvtColor(img, cv2.COLOR_BGR2HSV)
        elif mcolor == 'cielab':
            img = cv.cvtColor(img, cv2.COLOR_BGR2Lab)
        elif mcolor == 'ycbcr':
            img = cv.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        else:
            raise Exception('Correct values for color_models are : bgr, hsv, cielab, ycrcb')
        cropped = get_cropped(img, mask, points, axe1, axe2)
        images = cut_image(cropped)
        # features += f_color_mean(img, quadrants)
        # features += f_histo(img, quadrants)
        # features += f_distareas(img, quadrants, points)
        # features += list(f_entropy(cropped, quadrants))
        # features += f_superpixel_circles(img, center, quadrants)
        # features += f_lum_angles(img, quadrants, points)
        # features += f_superpixel_diff(images)
        # features = f_beta(img, mask)
        # features = f_beta_quad(images, mask)
    return features

# color_models : bgr, hsv, cielab, ycrcb


def load_data(folder, n_img=100, verbose=True, get_features=get_features, mask_folder=None):
    if n_img == -1:
        n_img = sum(len(files) for _, _, files in os.walk('/content/ISIC_2019/NON_SEGMENTEES/' + folder))
    lastVerbose = 0
    path = cfg.base + 'NON_SEGMENTEES/' + folder
    mask_path = cfg.base + 'SEGMENTEES/MASK/' + folder
    if mask_folder is not None:
        mask_path = cfg.base + 'SEGMENTEES/MASK/' + mask_folder
    X = []
    y = []
    for lesion_type in ('MEL', 'NEV'):
        current_path = path + lesion_type + '/'
        current_mask_path = mask_path + lesion_type + '/'
        files = sorted([f for f in listdir(current_path) if isfile(join(current_path, f))])
        i = 0
        for file in files:
            percent = round(i / n_img * 100 * 2)
            if i > n_img / 2:
                break
            if verbose and lastVerbose != percent:
                print(percent)
                lastVerbose = percent
                # print(current_path+file)
            img = cv2.imread(current_path + file)
            mask = cv2.imread(current_mask_path + file[:-4] + '_Mask.jpg', 0)
            img = cv2.resize(img, cfg.img_shape)
            mask = cv2.resize(mask, cfg.img_shape)
            X.append(get_features(img, mask))
            y.append(int(lesion_type == 'NEV'))
            i += 1
    return X, y


def load_all_data(folder, verbose=True, get_features=get_features, mask_folder=None):
    lastVerbose = 0
    path = cfg.base + 'NON_SEGMENTEES/' + folder
    mask_path = cfg.base + 'SEGMENTEES/MASK/' + folder
    if mask_folder is not None:
        mask_path = cfg.base + 'SEGMENTEES/MASK/' + mask_folder
    X = []
    y = []
    for lesion_type in ('MEL', 'NEV'):
        current_path = path + lesion_type + '/'
        current_mask_path = mask_path + lesion_type + '/'
        files = sorted([f for f in listdir(current_path) if isfile(join(current_path, f))])
        n_img = len(files)
        i = 0
        for file in files:
            percent = round(i / n_img * 100)
            if verbose and lastVerbose != percent:
                print(percent)
                lastVerbose = percent
                # print(current_path+file)
            img = cv2.imread(current_path + file)
            mask = cv2.imread(current_mask_path + file[:-4] + '_Mask.jpg', 0)
            img = cv2.resize(img, cfg.img_shape)
            mask = cv2.resize(mask, cfg.img_shape)
            X.append(get_features(img, mask))
            y.append(int(lesion_type == 'NEV'))
            i += 1
    return X, y
