import multiprocessing

n_cpu = multiprocessing.cpu_count()

PATH = "/home/adrien/ISIC_2019/NON_SEGMENTEES/"
GLOBAL_PATH = "/home/adrien/ISIC_2019/"

IMG_SIZE = 200
batch_size = 16
NUM_CLASSES = 2
epochs = 1
class_weight = {0: 1, 1: 4.6}
