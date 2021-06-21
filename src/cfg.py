import multiprocessing

n_cpu = multiprocessing.cpu_count()

# PATH = "/home/adrien/ISIC_2019/NON_SEGMENTEES/"
PATH = "/home/adrien/Subset_ISIC_2019/"
GLOBAL_PATH = "/home/adrien/ISIC_2019/"
FEATURES_PATH = "/home/adrien/Melanome/Features/"
WEIGHTS_PATH = "/home/adrien/Melanome/Weights/"

IMG_SIZE = 600
batch_size = 32
NUM_CLASSES = 2
epochs = 20
img_shape = (IMG_SIZE, IMG_SIZE, 3)
class_weight = {0: 1, 1: 4.59}
quad_rgb = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 0, 0)]
base = 'ISIC_2019/'
