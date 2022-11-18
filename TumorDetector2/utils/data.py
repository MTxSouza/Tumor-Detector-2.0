# imports
from sklearn.model_selection import train_test_split
from tensorflow._api.v2.data import TFRecordDataset
from typing import Union
from tqdm import tqdm
from PIL import Image

import tensorflow as tf
import pandas as pd
import numpy as np
import os



"""
This script allows to handle with image data in multiple ways for
tumor segmentation task.
"""


# variables
TFRECORD_READER = {
    'image': tf.io.FixedLenFeature(shape=[], dtype=tf.string),
    'mask': tf.io.FixedLenFeature(shape=[], dtype=tf.string),
    'label': tf.io.FixedLenFeature(shape=[], dtype=tf.int64)
}


"""
DataFrame methods
"""
def to_csv(image_path: str, mask_path: str) -> None:
    """Generate a CSV file which contains the path of image and it's respective mask.

    Args:
        image_path (str): Path to image folder.
        mask_path (str): Path to mask folder.
    """

    # checking variables
    assert isinstance(image_path, (str)), 'Invalid type. image_path must be a string'
    assert os.path.isdir(image_path), 'Invalid path. image_path must be a real folder path'
    
    assert isinstance(mask_path, (str)), 'Invalid type. mask_path must be a string'
    assert os.path.isdir(mask_path), 'Invalid path. mask_path must be a real folder path'

    # loading directories    
    __images = os.listdir(image_path)
    assert __images.__len__() == os.listdir(mask_path).__len__(), 'Invalid size. image folder has either more or less data than mask folder, both lengths must be the same'

    # building dataset structure
    __dataset = {
        'image': [],
        'mask': [],
        'label': []
    }
    
    # transforming data
    for current_image_name in tqdm(iterable=__images, desc='Transforming data'):
        
        # getting image and mask path
        __current_image_path = os.path.join(image_path, current_image_name)
        __current_mask_path = os.path.join(mask_path, current_image_name)
        
        # checking if both exists
        if os.path.exists(__current_image_path) and os.path.exists(__current_mask_path):
            
            for data_path in [__current_image_path, __current_mask_path]:
                
                # loading data and resizing it
                data = Image.open(fp=data_path)
                
                # checking mode and size
                if not data.mode == 'L':
                    data = data.convert(mode='L')
                if not data.size == (256,256):
                    data = data.resize(size=(256,256))
                    
                # updating image path with correct 
                # extension
                if not data_path.endswith('.jpeg'):
                    __new_data_path = data.replace('.' + data_path.split('.')[-1], '.jpeg')
                    data.save(fp=__new_data_path, format='JPEG')
                    data.close()
                    
                    # removing old data
                    os.remove(data_path)
                else:
                    # saving and closing data
                    data.save(fp=data_path, format='JPEG')
                    data.close()
    
    # verifying data    
    __images = os.listdir(image_path)
    for current_image_name in tqdm(iterable=__images, desc='Verifying data'):
        
        # getting image and mask path
        __current_image_path = os.path.join(image_path, current_image_name)
        __current_mask_path = os.path.join(mask_path, current_image_name)
        
        try:
            # checking if any data is corrupted
            for data_path in [__current_image_path, __current_mask_path]:
            
                __current_data = Image.open(fp=data_path)
                __current_data.verify()
                __current_data.close()
        except:
            for data_path in [__current_image_path, __current_mask_path]:
                os.remove(data_path)
        else:
            __dataset['image'].append(__current_image_path)
            __dataset['mask'].append(__current_mask_path)
            __dataset['label'].append(1 if np.array(Image.open(fp=__current_mask_path)).sum() else 0)
    
    # creating dataframe
    __dataset = pd.DataFrame(__dataset)
    
    # saving CSV
    __dataset.to_csv(path_or_buf='TumorDetector2/data/dataset.csv', encoding='utf-8')
    
    return pd.read_csv(filepath_or_buffer='TumorDetector2/data/dataset.csv', encoding='utf-8')

def split_dataset(dataset: pd.DataFrame, train_size: float, test_size: float) -> Union[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the CSV dataset and split it in three DataFrames.

    Args:
        dataset (pd.DataFrame): DataFrame storing the data path of image and mask and it's respected label.
        train_size (float): Percent representation of train size considering the entire dataset. Defaults to 0.7.
        test_size (float): Percent representation of test size after split in train data. Defaults to 0.3.

    Returns:
        Union[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Train, Test, Validation Dataframes.
    """
    
    # checking variables
    assert isinstance(dataset, (pd.DataFrame)), 'Invalid type. dataset must be a pd.DataFrame'
    
    assert isinstance(train_size, (float)), 'Invalid type. train_size must be a float'
    assert train_size > 0 and train_size <= 0.8, 'Invalid value. train_size cannot be equal to 0 or bigger than 0.8'
    
    assert isinstance(test_size, (float)), 'Invalid type. test_size must be a float'
    assert test_size > 0 and test_size <= 0.7, 'Invalid value. test_size cannot be equal to 0 or bigger than 0.7'
        
    # splitting into x and y
    __x, __y = dataset.iloc[:,:2], dataset.iloc[:,-1]
    
    # splitting into train, test, val
    __x_train, __x, __y_train, __y = train_test_split(__x, __y, train_size=train_size, shuffle=True, random_state=42)
    __x_val, __x_test, __y_val, __y_test = train_test_split(__x, __y, test_size=test_size, shuffle=True, random_state=42)
    
    # concating x and y
    __train = pd.concat([__x_train, __y_train], axis=1)
    __test = pd.concat([__x_test, __y_test], axis=1)
    __val = pd.concat([__x_val, __y_val], axis=1)
    
    return __train, __test, __val

"""
TFRecord methods
"""
def _tfrecord_reader(example_proto):
    
    # parsing example
    __content = tf.io.parse_single_example(serialized=example_proto, features=TFRECORD_READER)
    
    # getting data
    __image = __content['image']
    __mask = __content['mask']
    __label = __content['label']
    
    # parsing image tensors and reshaping them
    __image = tf.reshape(
        tensor=tf.io.parse_tensor(serialized=__image, out_type=tf.float32),
        shape=[256,256,1]
    )
    __mask = tf.reshape(
        tensor=tf.io.parse_tensor(serialized=__mask, out_type=tf.float32),
        shape=[256,256,1]
    )
    
    return __image, __mask, __label

def load_tfrecord(batch: int, prefetch: int, shuffle: bool) -> Union[TFRecordDataset, TFRecordDataset, TFRecordDataset]:
    """Read the TFRecord file and convert it into a Dataset.
    
    Args:
        batch (int): Batch size to split the dataset.
        prefetch (int): Number of data to be prefetch while the current batch is being processing.
        shuffle (bool): Shuffle data.
    
    Returns:
        Union[TFRecordDataset, TFRecordDataset, TFRecordDataset]: Train, Test, Validation Dataset.
    """
    
    # checking variables
    assert isinstance(batch, (int)), 'Invalid type. batch must be an int'
    assert batch > 0, 'Invalid value. batch must be bigger than zero'
    
    assert isinstance(prefetch, (int)), 'Invalid type. prefetch must be an int'
    assert prefetch > 0, 'Invalid value. prefetch must be bigger than zero'
    
    assert isinstance(shuffle, (bool)), 'Invalid type. shuffle must be a bool'
    
    # loading files
    __tfrecord_dataset = []
    
    for filename in ['train.tfrecord', 'test.tfrecord', 'val.tfrecord']:
        __tfrecord_dataset.append(
            TFRecordDataset(
                filenames=os.path.join('../data', filename)
            ).map(_tfrecord_reader).batch(batch).prefetch(prefetch).shuffle(shuffle)
        )
    
    return __tfrecord_dataset

def test_dataset() -> TFRecordDataset:
    """Load the test.tfrecord to test the trained model.
    *Make sure the run 'gen_tfrecord.py' to create
    'train.tfrecord', 'test.tfrecord' and 'val.tfrecord'
    first.
    """
    
    try:
        return tf.data.TFRecordDataset(filenames=os.path.join('TumorDetector2/data', 'test.tfrecord')).map(_tfrecord_reader).batch(1).shuffle(True)
    except Exception as e:
        raise FileExistsError(e)

"""
Features serialization
"""
def serialize_image(x, normalize):
    x = np.array(Image.open(x), dtype='float32')
    if normalize and x.max() == 255.0:
        x = x / 255.0 # diving by 255.0 to speed up training inference
    x = tf.convert_to_tensor(value=x, dtype=tf.float32)
    return tf.io.serialize_tensor(tensor=x)

def bytes_feature(x):
    if isinstance(x, type(tf.constant(0))):
        x = x.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[x]))

def int64_feature(x):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[x]))