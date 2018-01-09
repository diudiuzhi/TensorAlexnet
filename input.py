# coding=utf-8

import cPickle
import numpy as np
import tensorflow as tf

from config import get_conf


conf = get_conf()

# get configration
DATA_DIR = conf.get('data', 'dir')
TRAIN_SET_SIZE = int(conf.get('data', 'train_set'))
VALIDATION_SET_SIZE = int(conf.get('data', 'valiation_set'))
TEST_SET_SIZE = int(conf.get('data', 'test_set'))
IMAGE_SIZE = int(conf.get('data', 'image_size'))

TRAIN_IMAGES = []
TRAIN_LABELS = []

VALIDATION_IMAGES = []
VALIDATION_LABELS = []

TEST_IMAGES = []
TEST_LABELS = []


def unpickle(data_dir):
    with open(data_dir, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict


def _distorted_image(image):
    height = IMAGE_SIZE
    width = IMAGE_SIZE
    
    reshaped_image = tf.cast(image, tf.float32)
    distorted_image = tf.random_crop(reshaped_image, [height, width, 3])
    distorted_image = tf.image.random_flip_left_right(distorted_image)
    distorted_image = tf.image.random_brightness(distorted_image,
                                               max_delta=63)
    distorted_image = tf.image.random_contrast(distorted_image,
                                             lower=0.2, upper=1.8)
    
    float_image = tf.image.per_image_standardization(distorted_image)
    float_image.set_shape([height, width, 3])
    
    return float_image


def get_train_batch_data(batch_size):
    if batch_size <= 0:
        raise ValueError("batch size should greater than 0")
     
    valuequeue = tf.train.input_producer(TRAIN_IMAGES, shuffle=False)
    valuelabel = tf.train.input_producer(TRAIN_LABELS, shuffle=False)
       
    image = valuequeue.dequeue()
    label = valuelabel.dequeue()
    
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, [3, 32, 32])
    image = tf.transpose(image, [1, 2, 0])
    
    label = tf.cast(label, tf.int32)
    
    float_image = _distorted_image(image)
    
    min_queue_examples = int(batch_size * 0.4)
    
    images, labels = tf.train.batch(
        [float_image, label],
        batch_size=batch_size,
        num_threads=16,
        capacity=min_queue_examples + 3 * batch_size)
    
    return images, labels
    
    
def get_validation_batch_data(batch_size):
    if batch_size <= 0:
        raise ValueError("batch size should greater than 0")
     
    valuequeue = tf.train.input_producer(VALIDATION_IMAGES, shuffle=False)
    valuelabel = tf.train.input_producer(VALIDATION_LABELS, shuffle=False)
       
    image = valuequeue.dequeue()
    label = valuelabel.dequeue()
    
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, [3, 32, 32])
    image = tf.transpose(image, [1, 2, 0])
    
    label = tf.cast(label, tf.int32)
    float_image = _distorted_image(image)
    
    min_queue_examples = int(batch_size * 0.4)
    
    images, labels = tf.train.batch(
        [float_image, label],
        batch_size=batch_size,
        num_threads=16,
        capacity=min_queue_examples + 3 * batch_size)
    
    return images, labels


def get_test_batch_data(batch_size):
    if batch_size <= 0:
        raise ValueError("batch size should greater than 0")
     
    valuequeue = tf.train.input_producer(TEST_IMAGES, shuffle=False)
    valuelabel = tf.train.input_producer(TEST_LABELS, shuffle=False)
       
    image = valuequeue.dequeue()
    label = valuelabel.dequeue()
    
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, [3, 32, 32])
    image = tf.transpose(image, [1, 2, 0])
    
    label = tf.cast(label, tf.int32)
    float_image = _distorted_image(image)
    
    min_queue_examples = int(batch_size * 0.4)
    
    images, labels = tf.train.batch(
        [float_image, label],
        batch_size=batch_size,
        num_threads=16,
        capacity=min_queue_examples + 3 * batch_size)
    
    return images, labels


def get_data_from_file():
    # train and validation data
    t_v_datas, t_v_labels = _get_train_and_validation_data()
     
    # test data
    t_datas, t_labels = _get_test_data()
    
    # get train dataset
    global TRAIN_IMAGES; TRAIN_IMAGES = t_v_datas[:TRAIN_SET_SIZE]
    global TRAIN_LABELS; TRAIN_LABELS = t_v_labels[:TRAIN_SET_SIZE]
    
    # get validation dataset
    global VALIDATION_IMAGES; VALIDATION_IMAGES = t_v_datas[TRAIN_SET_SIZE:]
    global VALIDATION_LABELS; VALIDATION_LABELS = t_v_labels[TRAIN_SET_SIZE:]
    
    # get test dataset
    global TEST_IMAGES; TEST_IMAGES = t_datas[:TEST_SET_SIZE]
    global TEST_LABELS; TEST_LABELS = t_labels[:TEST_SET_SIZE]
    

def _get_train_and_validation_data():
    datas = unpickle(DATA_DIR + 'data_batch_1')
    t_v_datas = datas['data']
    t_v_labels = datas['labels']
    
    for i in range(2,6):
        datas = unpickle(DATA_DIR + 'data_batch_' + str(i))
        images = datas['data']
        labels = datas['labels']
    
        t_v_datas = np.vstack((t_v_datas, images))
        t_v_labels = t_v_labels + labels
    
    return t_v_datas, t_v_labels
    

def _get_test_data():
    datas = unpickle(DATA_DIR + 'test_batch')
    images = datas['data']
    labels = datas['labels']
    
    return images, labels


get_data_from_file()    
