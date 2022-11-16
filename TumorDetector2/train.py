# imports
from TumorDetector2.utils.weights import load_architecture,\
                                         TrainLogging
from TumorDetector2.utils.metrics import DiceCoeficient,\
                                         IoU,\
                                         loss_menager,\
                                         apply_threshold
from tensorflow.keras.losses import BinaryCrossentropy
from TumorDetector2.utils.data import load_tfrecord
from tensorflow.keras.optimizers import Adam
from tensorflow import GradientTape
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
    EPOCHS = arg.epochs
    BATCH = arg.batch
    LOSS = BinaryCrossentropy()
    METRIC = arg.metrics
    SHUFFLE = arg.shuffle
    LEARNING_RATE = arg.learning_rate
    OPTIMIZER = Adam(LEARNING_RATE)
    PREFETCH = arg.prefetch
    THRESHOLD = arg.threshold
    EARLY_STOP = arg.early_stop
    LOSS_TRANSFORMER = arg.loss_transformer

    # checking variables
    assert isinstance(EPOCHS, (int)), 'Invalid type. --epochs must be an int'
    assert EPOCHS > 0, 'Invalid value. --epochs must be bigger than zero'

    assert isinstance(METRIC, (str)), 'Invalid type. --metrics must be a str'
    assert METRIC in [IoU.name, DiceCoeficient.name], 'Invalid value. --metrics must be one of them: [intersection over union, dice coeficient]'

    if METRIC == 'intersection over union':
        METRIC = IoU()
    elif METRIC == 'dice coeficient':
        METRIC = DiceCoeficient()

    assert isinstance(LEARNING_RATE, (float)), 'Invalid type. --learning-rate must be a float'
    assert LEARNING_RATE > 0, 'Invalid value. --learining-rate must be bigger than zero'

    assert isinstance(THRESHOLD, (float)), 'Invalid type. --threshold must be a float'
    assert THRESHOLD >= 0, 'Invalid value. --threshold must be bigger than zero or equal'

    assert isinstance(EARLY_STOP, (int)), 'Invalid type. --early-stop must be an int'
    assert EARLY_STOP > 0, 'Invalid value. --early-stop must be bigger than zero'

    # loading model
    MODEL = load_architecture(version=VERSION)

    # loading tfrecords
    TRAIN_DATASET, TEST_DATASET, VAL_DATASET = load_tfrecord(batch=BATCH, prefetch=PREFETCH, shuffle=SHUFFLE)

    # training
    LOSSES = []
    VAL_LOSSES = []

    ACCURACIES = []
    VAL_ACCURACIES = []

    BEST_LOSS = None
    BEST_VAL_LOSS = None

    ACCURACY = None
    VAL_ACCURACY = None

    PREVIOUS_VAL_LOSS = None

    OVERFITTING = 0

    TOTAL_ITERATIONS = None

    TRAIN_LOGGING = TrainLogging()

    print('-'*80)
    print('Tumor Detector 2.0 - BRAIN (Brazilian Artificial Inteligence Nucleus)')
    print('-'*80)
    print(f'Loss Function: {LOSS.name} | Accuracy function: {METRIC.name} | Optimizer: {OPTIMIZER._name}')
    print(f'Learning Rate: {LEARNING_RATE} - Batch size: {BATCH} - Epochs: {EPOCHS} - Threshold: {THRESHOLD} - Early Stop: {EARLY_STOP}')
    print('-'*80)

    for epoch in range(1, EPOCHS+1):
    
        # training inference
        ITER_TRAINING = tqdm(iterable=TRAIN_DATASET, desc=f'EPOCH: {epoch}/{EPOCHS} | Training', total=TOTAL_ITERATIONS)
        
        mean_loss = 0
        mean_accuracy = 0
        for iterations, (image, mask, label) in enumerate(ITER_TRAINING):
            
            with GradientTape() as tape:
            
                # running inference
                predicted = MODEL(image, training=True)
                
                # calculating loss
                loss = loss_menager(
                    loss=LOSS(mask, predicted),
                    apply=LOSS_TRANSFORMER
                )
                
                # applying threshold
                if THRESHOLD != 0.0 and type(THRESHOLD) == float:
                    predicted = apply_threshold(predicted, THRESHOLD)
                
                # calculating accuracy
                accuracy = METRIC(mask, predicted)
                
                # saving results
                mean_loss += loss
                mean_accuracy += accuracy
                
                # gradiant
                grads = tape.gradient(target=loss, sources=MODEL.trainable_weights)
                OPTIMIZER.apply_gradients(grads_and_vars=zip(grads, MODEL.trainable_weights))
        mean_loss /= (iterations + 1) # mean loss of training
        mean_accuracy /= (iterations + 1) # mean accuracy of training
        
        # saving total iterations
        if TOTAL_ITERATIONS is None:
            TOTAL_ITERATIONS = iterations + 1
        
        # validating
        print('Validating..')
        
        mean_val_loss = 0
        mean_val_accuracy = 0
        SAMPLES = []
        for iterations, (image, mask, label) in enumerate(VAL_DATASET):
            
            # running inference
            predicted = MODEL(image, training=False)
            
            # calculating loss
            loss = loss_menager(
                    loss=LOSS(mask, predicted),
                    apply=LOSS_TRANSFORMER
                )

            # calculating accuracy
            if THRESHOLD != 0.0 and type(THRESHOLD) == float:
                accuracy = METRIC(mask, apply_threshold(predicted, THRESHOLD))
            else:
                accuracy = METRIC(mask, predicted)
            
            # saving results
            mean_val_loss += loss
            mean_val_accuracy += accuracy
        mean_val_loss /= (iterations + 1) # mean loss of validation
        mean_val_accuracy /= (iterations + 1) # mean accuracy of validation

        # saving weights
        SAVE = False
        if BEST_LOSS is None or (mean_loss < BEST_LOSS and mean_val_loss < BEST_VAL_LOSS):
            BEST_LOSS = mean_loss
            BEST_VAL_LOSS = mean_val_loss
            ACCURACY = mean_accuracy
            VAL_ACCURACY = mean_val_accuracy
            SAVE = True

        TRAIN_LOGGING.save(
            MODEL,
            VERSION,
            BEST_LOSS,
            BEST_VAL_LOSS,
            ACCURACY,
            VAL_ACCURACY,
            LOSS.name,
            METRIC.name,
            OPTIMIZER._name,
            epoch,
            EPOCHS,
            THRESHOLD,
            LEARNING_RATE,
            BATCH,
            LOSS_TRANSFORMER,
            None,
            None,
            SAVE
        )
        
        # displaying results
        print(f'| Train loss: {mean_loss.numpy()} - Val loss: {mean_val_loss.numpy()} | Train accuracy: {mean_accuracy.numpy()} - Val accuracy: {mean_val_accuracy.numpy()} |')
        print('-'*80)