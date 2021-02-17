import multiprocessing

n_cpu = multiprocessing.cpu_count()

#PATH = "/home/adrien/ISIC_2019/NON_SEGMENTEES/"
PATH = "/home/adrien/Subset_ISIC_2019/"
GLOBAL_PATH = "/home/adrien/ISIC_2019/"

IMG_SIZE = 200
batch_size = 16
NUM_CLASSES = 2
epochs = 5
#class_weight = {0: 1, 1: 4.6}
class_weight = {0: 1, 1: 1}
