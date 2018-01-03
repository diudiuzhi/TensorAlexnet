# coding=utf-8

import tensorflow as tf

from config import get_conf
import input 

conf = get_conf()

TRAIN_SET_SIZE = int(conf.get('data', 'train_set'))
BATCH_SIZE = int(conf.get('train', 'batch_size'))
NUM_EPOCHS_PER_DECAY = float(conf.get('train', 'num_epochs_per_decay'))
INITIAL_LEARNING_RATE = float(conf.get('train', 'initial_learning_rate'))
LEARNING_RATE_DECAY_FACTOR = float(conf.get('train', 'learning_rate_decay_factor'))

MOMENTUM = float(conf.get('train', 'momentum'))
EPOCH_NUM = int(conf.get('train', 'epoch_num'))

DROPOUT = float(conf.get('train', 'dropout'))


def conv2d(_x, _w, _b):
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(_x, _w, [1, 1, 1, 1], padding='SAME'), _b))


def max_pool(_x, f):
    return tf.nn.max_pool(_x, [1, f, f, 1], [1, 1, 1, 1], padding='SAME')


def lrn(_x):
    return tf.nn.lrn(_x, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)


def inference(images, parameters):
    '''Build the network model and return logits'''
    # conv1
    conv1 = conv2d(images, parameters['w1'], parameters['bw1'])
    lrn1 = lrn(conv1)
    pool1 = max_pool(lrn1, 2)
    
    # conv2
    conv2 = conv2d(pool1, parameters['w2'], parameters['bw2'])
    lrn2 = lrn(conv2)
    pool2 = max_pool(lrn2, 2)
    
    # conv3
    conv3 = conv2d(pool2, parameters['w3'], parameters['bw3'])
    
    # conv4
    conv4 = conv2d(conv3, parameters['w4'], parameters['bw4'])
    
    # conv5
    conv5 = conv2d(conv4, parameters['w5'], parameters['bw5'])
    pool5 = max_pool(conv5, 2)
    
    # FC1
    shape = pool5.get_shape()
    reshape = tf.reshape(pool5, [-1, shape[1].value*shape[2].value*shape[3].value])
    fc1 = tf.nn.relu(tf.matmul(reshape, parameters['fc1']) + parameters['bc1'])
    fc1_drop = tf.nn.dropout(fc1, keep_prob=DROPOUT)
    
    # FC2
    fc2 = tf.nn.relu(tf.matmul(fc1_drop, parameters['fc2']) + parameters['bc2'])
    fc2_drop = tf.nn.dropout(fc2, keep_prob=DROPOUT)
    
    softmax_linear = tf.add(tf.matmul(fc2_drop, parameters['softmax']), parameters['bs'])
    
    return softmax_linear


def loss_function(logits, labels):
    '''return loss'''
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    
    return cross_entropy_mean


def train_step(loss, global_step):
    num_batches_per_epoch = TRAIN_SET_SIZE / BATCH_SIZE
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
    
    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                  global_step,
                                  decay_steps,
                                  LEARNING_RATE_DECAY_FACTOR,
                                  staircase=True)
    
    train_op = tf.train.MomentumOptimizer(lr, MOMENTUM).minimize(loss)
    return train_op
    

def train():
    with tf.Graph().as_default():
        parameters = {
            'w1': tf.Variable(tf.truncated_normal([3, 3, 3, 24], dtype=tf.float32, stddev=0.055), name='w1'),
            'w2': tf.Variable(tf.truncated_normal([3, 3, 24, 96], dtype=tf.float32, stddev=0.0098), name='w2'),
            'w3': tf.Variable(tf.truncated_normal([3, 3, 96, 192], dtype=tf.float32, stddev=0.0034), name='w3'),
            'w4': tf.Variable(tf.truncated_normal([3, 3, 192, 192], dtype=tf.float32, stddev=0.0024), name='w4'),
            'w5': tf.Variable(tf.truncated_normal([3, 3, 192, 96], dtype=tf.float32, stddev=0.0034), name='w5'),
            'fc1': tf.Variable(tf.truncated_normal([96*32*32, 1024], dtype=tf.float32, stddev=1e-2), name='fc1'),
            'fc2': tf.Variable(tf.truncated_normal([1024, 1024], dtype=tf.float32, stddev=1e-2), name='fc2'),
            'softmax': tf.Variable(tf.truncated_normal([1024, 10], dtype=tf.float32, stddev=1e-2), name='fc3'),

            'bw1': tf.Variable(tf.random_normal([24])),
            'bw2': tf.Variable(tf.random_normal([96])),
            'bw3': tf.Variable(tf.random_normal([192])),
            'bw4': tf.Variable(tf.random_normal([192])),
            'bw5': tf.Variable(tf.random_normal([96])),
            'bc1': tf.Variable(tf.random_normal([1024])),
            'bc2': tf.Variable(tf.random_normal([1024])),
            'bs': tf.Variable(tf.random_normal([10]))
        }
        
        global_step = tf.train.get_or_create_global_step()
        
        with tf.device('/cpu:0'):
            images, labels = input.get_train_batch_data(BATCH_SIZE)
          
        # train step
        logits = inference(images, parameters)
        loss = loss_function(logits, labels)
        train_op = train_step(loss, global_step)
        
        # validation step
        
        #with tf.device('/cpu:0'):
        #    validation_images, validation_labels = input.get_validation_batch_data()
        #accuracy = inference(images, parameters)
        
        
        with tf.train.MonitoredTrainingSession(
            hooks=[tf.train.StopAtStepHook(last_step=EPOCH_NUM)]) as mon_sess:
            
            while not mon_sess.should_stop():
                loss = mon_sess.run([loss,train_op])
                print("%d : %f" % (global_step, loss))
                
                
def main():
    train()
  

if __name__ == "__main__":
    main()
    