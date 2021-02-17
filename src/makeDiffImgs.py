import sys

from skimage import img_as_float, io
from features import get_diff_img
import cfg
from os import listdir
from os.path import isfile, join
import numpy as np


outpath = "/home/adrien/Subset_ISIC_2019/Diff_images/"

for folder in ("TRAIN", "TEST"):
    print("Loading " + folder + " data...")
    for lesion_type in ("NEV", "MEL"):
        print("Loading " + lesion_type + " data...")
        current_path = cfg.PATH + folder + "/" + lesion_type + "/"
        files = [f for f in listdir(current_path) if isfile(join(current_path, f))]
        i = 0
        for img in files:
            image = img_as_float(io.imread(current_path + img))
            diff = (get_diff_img(mask,image)*255).astype(np.uint8)
            io.imsave(outpath+folder+'/'+img, diff)
            i += 1

            sys.stdout.write('\r')
            sys.stdout.write(str(round(i / len(files)*100)) + "%")
print("\nDone!")
