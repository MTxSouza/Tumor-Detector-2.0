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
    parser.add_argument('--epochs', '-e', type=int, help='Number of iterations to be done while training.')
    parser.add_argument('--batch', '-b', type=int, help='Number of data to be present in each batch (block of data used to run each inference).')
    parser.add_argument('--metrics', '-m', type=str, default='dice coeficient', help='Accuracy metrics to use.')
    parser.add_argument('--learning-rate', '-lr', type=float, default=1e-5, help='Learning rate used to update the weights.')
    parser.add_argument('--threshold', '-th', type=float, default=0.5, help='Threshold used to limit what will be considered either a class or another.')
    arg = parser.parse_args()


if __name__=='__main__':

    pass