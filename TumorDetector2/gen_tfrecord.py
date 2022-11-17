# imports
from TumorDetector2.utils.data import to_csv,\
                                      serialize_image,\
                                      bytes_feature,\
                                      int64_feature,\
                                      split_dataset,\
                                      load_tfrecord
from tensorflow._api.v2.io import TFRecordWriter                
from tqdm import tqdm

import tensorflow as tf
import pandas as pd
import numpy as np
import argparse
import os


"""
This script allows to create TFRecord files of Tumor Detector 2.0 project.
"""



def __initial_parameters() -> None:
    
    global arg
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', '-i', type=str, help='Path to image dataset folder.')
    parser.add_argument('--mask', '-m', type=str, help='Path to mask dataset folder.')
    parser.add_argument('--train-size', '-train', type=float, default=0.7, help='Percentual representation of data to be used for training.')
    parser.add_argument('--test-size', '-test', type=float, default=0.1, help='Percentual representation of remaining data to be used for testing.')
    parser.add_argument('--ignore-csv', '-skip', action='store_false', help='Force to create the tfrecords with an exists CSV file.')
    arg = parser.parse_args()


if __name__=='__main__':
    
    # getting initial arguments
    __initial_parameters()
    
    IMAGE_PATH = arg.image
    MASK_PATH = arg.mask
    TRAIN_SIZE = arg.train_size
    TEST_SIZE = arg.test_size
    IGNORE_CSV = arg.ignore_csv
    
    # checking if dataset.csv already exists to load it
    # or to create a new one
    if not IGNORE_CSV:
        try:
            __dataset = pd.read_csv('TumorDetector2/data/dataset.csv')
            # removing 'Unnamed: 0' column if it exists
            if 'Unnamed: 0' in __dataset.columns:
                __dataset.drop(labels='Unnamed: 0', axis=1, inplace=True)
        except Exception as e:
            print(f'Could not load an exists CSV dataset file. {e}')
    else:
        __dataset = to_csv(IMAGE_PATH, MASK_PATH)
        
    # removing 'Unnamed: 0' column
    if 'Unnamed: 0' in __dataset.columns:
        __dataset.drop(labels='Unnamed: 0', axis=1, inplace=True)
        
    # splitting dataset into train, test and validation
    # data
    TRAIN, TEST, VAL = split_dataset(__dataset, TRAIN_SIZE, TEST_SIZE)
    
    # writting tfrecord file
    for dataframe, filename, normalize in zip([TRAIN, TEST, VAL], ['train.tfrecord', 'test.ftrecord', 'val.tfrecord'], [True, False, False]):
        
        with TFRecordWriter(path=os.path.join('TumorDetector2/data', filename)) as tfrecord:
            for image, mask, label in tqdm(iterable=np.array(dataframe), desc=f'Writting {filename}'):
                
                # creating tfrecord example
                __example = tf.train.Example(features=tf.train.Features(
                    feature={
                        'image': bytes_feature(x=serialize_image(image, normalize)),
                        'mask': bytes_feature(x=serialize_image(mask, True)),
                        'label': int64_feature(x=label)
                    }
                ))
                tfrecord.write(__example.SerializeToString())
                
    # checking tfrecord files
    TRAIN, TEST, VAL = load_tfrecord(2, 1, True)
    
    try:
        for (image, mask, label) in TEST:
            pass
    except Exception as e:
        raise RuntimeError(e)
            
    print('TFRecord has been created at Tumor-Detector-2.0/TumorDetector2/data/')
