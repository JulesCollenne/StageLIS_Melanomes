import cv2
import sys

import cfg
from visu import draw_quadrants_on_lesion

if len(sys.argv) < 2:
    img_name = "ISIC_0000082_downsampled"
else:
    img_name = sys.argv[1][:-4]


mask_path = cfg.GLOBAL_PATH+"SEGMENTEES/MASK/TRAIN/NEV/"+img_name+"_Mask.jpg"
lesion_path = cfg.GLOBAL_PATH+"NON_SEGMENTEES/TRAIN/NEV/"+img_name+".JPG"
mask = cv2.imread(mask_path, 0)
lesion = cv2.imread(lesion_path)

if mask is None:
    print("Fichier introuvable !")
    print(mask_path)
    exit(1)
if lesion is None:
    print("Fichier introuvable !")
    print(mask_path)
    exit(1)

draw_quadrants_on_lesion(lesion, mask, name=img_name+".jpg")
