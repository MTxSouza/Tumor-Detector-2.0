# imports
from TumorDetector2.utils.weights import load_architecture
from TumorDetector2.utils.metrics import DiceCoeficient,\
                                         IoU,\
                                         loss_menager,\
                                         apply_threshold
from TumorDetector2.utils.data import load_tfrecord
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm

import argparse



def __initial_parameters():

    global arg

    parser = argparse.ArgumentParser()
    parser.add_argument('--version', '-v', type=str, help='Model version to use.')
    parser.add_argument('--epochs', '-e', type=int, help='Number of iterations to be done while training.')
    parser.add_argument('--batch', '-b', type=int, help='Number of data to be present in each batch (block of data used to run each inference).')
    parser.add_argument('--metrics', '-m', type=str, default='dice coeficient', help='Accuracy metrics to use.')
    parser.add_argument('--learning-rate', '-lr', type=float, default=1e-5, help='Learning rate used to update the weights.')
    parser.add_argument('--shuffle', 'sh', action='store_true', help='Shuffle the data before start training.')
    parser.add_argument('--prefetch', '-pre', type=int, default=2, help='Specify the number of data to be preprocesed while a current data is being calculated.')
    parser.add_argument('--threshold', '-th', type=float, default=0.5, help='Threshold used to limit what will be considered either a class or another.')
    parser.add_argument('--early-stop', '-stop', type=int, default=5, help='Force to stop training if start overfitting.')
    parser.add_argument('--loss-transformer', '-lt', type=str, default='real', help='Specify the type of transformation to be done in the loss value.')
    arg = parser.parse_args()


if __name__=='__main__':

    # getting initial arguments
    __initial_parameters()

    VERSION = arg.version
    EPOCH = arg.epochs
    BATCH = arg.batch
    METRICS = arg.metrics
    SHUFFLE = arg.shuffle
    LEARINING_RATE = arg.learning_rate
    PREFETCH = arg.prefetch
    THRESHOLD = arg.threshold
    EARLY_STOP = arg.early_stop
    LOSS_TRANSFORMER = arg.loss_transformer

    # checking variables
    assert isinstance(EPOCH, (int)), 'Invalid type. --epochs must be an int'
    assert EPOCH > 0, 'Invalid value. --epochs must be bigger than zero'

    # loading model
    MODEL = load_architecture(version=VERSION)

    # loading tfrecords
    TRAIN, TEST, VAL = load_tfrecord(batch=BATCH, prefetch=PREFETCH, shuffle=SHUFFLE)