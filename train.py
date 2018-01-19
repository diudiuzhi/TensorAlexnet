# coding=utf-8

import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages
import time

from config import get_conf
import input 

conf = get_conf()

TRAIN_SET_SIZE = int(conf.get('data', 'train_set'))
BATCH_SIZE = int(conf.get('train', 'batch_size'))
DECAY_STEP = float(conf.get('train', 'decay_step'))
INITIAL_LEARNING_RATE = float(conf.get('train', 'initial_learning_rate'))
LEARNING_RATE_DECAY_FACTOR = float(conf.get('train', 'learning_rate_decay_factor'))

MOMENTUM = float(conf.get('train', 'momentum'))
EPOCH_NUM = int(conf.get('train', 'epoch_num'))

DROPOUT = float(conf.get('train', 'dropout'))


def conv2d(_x, _w, _b):
    return tf.nn.bias_add(tf.nn.conv2d(_x, _w, [1, 1, 1, 1], padding='SAME'), _b)


def max_pool(_x, f):
    return tf.nn.max_pool(_x, [1, f, f, 1], [1, 1, 1, 1], padding='SAME')


def lrn(_x):
    return tf.nn.lrn(_x, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)


def init_w(namespace, shape, wd, stddev, reuse=False):
    with tf.variable_scope(namespace, reuse=reuse):
        initializer = tf.truncated_normal_initializer(dtype=tf.float32, stddev=stddev)
        w = tf.get_variable("w", shape=shape, initializer=initializer)
        
        if wd:
            weight_decay = tf.multiply(tf.nn.l2_loss(w), wd, name='weight_loss')
            tf.add_to_collection('losses', weight_decay)
    return w


def init_b(namespace, shape, reuse=False):
    with tf.variable_scope(namespace, reuse=reuse):
        initializer = tf.constant_initializer(0.0)
        b = tf.get_variable("b", shape=shape, initializer=initializer)
    return b


def batch_normal(xs, out_size):
    x_shape = x.get_shape()
    params_shape = x_shape[-1:]
    
    axis = list(range(len(xs.get_shape()) - 1))
    
    beta = tf.get_variable('beta', params_shape, initializer=tf.zeros_initializer)
    gamma = tf.get_variable('gamma', params_shape, initializer=tf.ones_initializer)
    
    moving_mean = tf.get_variable('moving_mean', params_shape, initializer=tf.zeros_initializer, trainable=False)
    moving_variance = tf.get_variable('moving_variance', params_shape, initializer=tf.ones_initializer, trainable=False)
    
    mean, variance = tf.nn.moments(xs, axes=axis)
    update_moving_mean = moving_averages.assign_moving_average(
        moving_mean, mean, 0.9997)
    
    update_moving_variance = moving_averages.assign_moving_average(
        moving_variance, variance, 0.9997)
    
    tf.add_to_collection("resnet_update_ops", update_moving_mean)
    tf.add_to_collection("resnet_update_ops", update_moving_variance)
    
    mean, variance = control_flow_ops.cond(
        True, lambda: (mean, variance),
        lambda: (moving_mean, moving_variance))
    
    bn = tf.nn.batch_normalization(xs, mean, variance, beta, gamma, 0.001)
    return bn
    

def inference(images, reuse=False):
    '''Build the network model and return logits'''
    
    # conv1
    w1 = init_w("conv1", [3, 3, 3, 24], None, 0.01, reuse)
    bw1 = init_b("conv1", [24], reuse)
    conv1 = conv2d(images, w1, bw1)
    bn1 = batch_normal(conv1, 24)
    c_output1 = tf.nn.relu(bn1)
    pool1 = max_pool(c_output1, 2)
    
    # conv2
    w2 = init_w("conv2", [3, 3, 24, 96], None, 0.01, reuse)
    bw2 = init_b("conv2", [96], reuse)
    conv2 = conv2d(pool1, w2, bw2)
    bn2 = batch_normal(conv2, 96)
    c_output2 = tf.nn.relu(bn2)
    pool2 = max_pool(c_output2, 2)
    
    # conv3
    w3 = init_w("conv3", [3, 3, 96, 192], None, 0.01, reuse)
    bw3 = init_b("conv3", [192], reuse)
    conv3 = conv2d(pool2, w3, bw3)
    bn3 = batch_normal(conv3, 192)
    c_output3 = tf.nn.relu(bn3)
    pool3 = max_pool(c_output3, 2)
    
                
    # FC1
    wfc1 = init_w("fc1", [192*24*24, 1024], None, 1e-2, reuse)
    bfc1 = init_b("fc1", [1024], reuse)
    shape = pool3.get_shape()
    reshape = tf.reshape(pool3, [-1, shape[1].value*shape[2].value*shape[3].value])
    w_x1 = tf.matmul(reshape, wfc1) + bfc1
    bn4 = batch_normal(w_x1, 1024)
    fc1 = tf.nn.relu(bn4)
    
    # FC2
    wfc2 = init_w("fc2", [1024, 10], None, 1e-2, reuse)
    bfc2 = init_b("fc2", [10], reuse)
    softmax_linear = tf.add(tf.matmul(fc1, wfc2), bfc2)
    bn5 = batch_normal(softmax_linear, 10)
    return bn5


def loss_function(logits, labels):
    '''return loss'''
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    tf.add_to_collection('losses', cross_entropy_mean)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def train_step(loss, global_step):
    
    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                  global_step,
                                  DECAY_STEP,
                                  LEARNING_RATE_DECAY_FACTOR,
                                  staircase=True)
    tf.add_to_collection('learning_rate', lr)
    
    train_op = tf.train.MomentumOptimizer(lr, MOMENTUM).minimize(loss)
    return train_op
    

def train():
    with tf.Graph().as_default():
        
        global_step = tf.train.get_or_create_global_step()
        
        with tf.device('/cpu:0'):
            images, labels = input.get_train_batch_data(BATCH_SIZE)
          
        # train step
        logits = inference(images)
        loss = loss_function(logits, labels)
        train_op = train_step(loss, global_step)
        
        train_labels = tf.one_hot(labels, depth=10)
        train_correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(train_labels, 1))
        train_accuracy = tf.reduce_mean(tf.cast(train_correct_pred, tf.float32))
        
        ##### Test step
        with tf.device('/cpu:0'):
            test_images, test_labels = input.get_test_batch_data(BATCH_SIZE)
        test_logits = inference(test_images, True)
        test_labels = tf.one_hot(test_labels, depth=10)
        
        test_correct_pred = tf.equal(tf.argmax(test_logits, 1), tf.argmax(test_labels, 1))
        test_accuracy = tf.reduce_mean(tf.cast(test_correct_pred, tf.float32))
        
        add_global = global_step.assign_add(1)
        
        with tf.train.MonitoredTrainingSession(
            hooks=[tf.train.StopAtStepHook(last_step=EPOCH_NUM)]) as mon_sess:
            
            f = open("result.txt", 'a+')
            
            train_acc = 0.0
            while not mon_sess.should_stop():
                mon_sess.run(train_op)
                train_acc += mon_sess.run(train_accuracy)
                
                step = mon_sess.run(add_global)
                
                if step % 195 == 0:
                    lo =  mon_sess.run(loss)
                    lr = mon_sess.run(tf.get_collection('learning_rate'))
                    
                    print("%d  losses: %f" % (step, lo))
                    print("%d  learning rate: %f" % (step, lr[0]))
                    f.write("%.5f\n" % lo)
                    f.write("%.5f\n" % lr[0])
                    
                    train_acc /= 195
                    print("%d  Train acc: %f" % (step, train_acc))
                    f.write("%.5f\n" % train_acc)
                    f.flush()
                    train_acc = 0.0
                    
                    test_acc = 0.0
                    test_epoch = int(10000/BATCH_SIZE)
                    
                    for i in range(test_epoch):
                        test_acc += mon_sess.run(test_accuracy)

                    test_acc /= test_epoch
                    
                    print("%d  Test acc: %f" % (step, test_acc))
                    f.write("%.5f\n" % test_acc)
                    f.flush()
                    
            print("Train over")
                
def main():
    train()
  

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print("Total time: %f" % (end_time-start_time))
    