import getopt
import sys
import tensorflow as tf

from dataset import get_datasets
from models import get_efficientnet


def main(argv):
    model_name = 'b2'

    try:
        opts, args = getopt.getopt(argv, "hi:o:", ["ifile=", "ofile="])
    except getopt.GetoptError:
        print('make_predictions.py -m <model_name> -o <outputfile>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('test.py -i <inputfile> -o <outputfile>')
            sys.exit()
        elif opt in ("-m", "--model"):
            model_name = opt
        elif opt in ("-o", "--ofile"):
            outputfile = arg

    train_ds, val_ds, test_ds = get_datasets()
    sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
    weights_path = '/content/drive/MyDrive/Stage_LIS/Weights/'
    b2 = get_efficientnet(model_name)
    b2.load_weights(weights_path + model_name + '_fine.h5')


if __name__ == "__main__":
    main(sys.argv[1:])
