# coding=utf-8

import tensorflow as tf

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
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(_x, _w, [1, 1, 1, 1], padding='SAME'), _b))


def max_pool(_x, f):
    return tf.nn.max_pool(_x, [1, f, f, 1], [1, 1, 1, 1], padding='SAME')


def lrn(_x):
    return tf.nn.lrn(_x, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)


def init_w(namespace, shape, stddev, reuse=False):
    with tf.variable_scope(namespace, reuse=reuse):
        initializer = tf.truncated_normal_initializer(dtype=tf.float32, stddev=stddev)
        w = tf.get_variable("w", shape=shape, initializer=initializer)
    return w


def init_b(namespace, shape, reuse=False):
    with tf.variable_scope(namespace, reuse=reuse):
        initializer = tf.constant_initializer(0.0)
        b = tf.get_variable("b", shape=shape, initializer=initializer)
    return b


def inference(images, reuse=False):
    '''Build the network model and return logits'''
    # conv1
    w1 = init_w("conv1", [3, 3, 3, 24], 0.055, reuse)
    bw1 = init_b("conv1", [24], reuse)
    conv1 = conv2d(images, w1, bw1)
    lrn1 = lrn(conv1)
    pool1 = max_pool(lrn1, 2)
    
    # conv2
    w2 = init_w("conv2", [3, 3, 24, 96], 0.0034, reuse)
    bw2 = init_b("conv2", [96], reuse)
    conv2 = conv2d(pool1, w2, bw2)
    lrn2 = lrn(conv2)
    pool2 = max_pool(lrn2, 2)
    
    # conv3
    w3 = init_w("conv3", [3, 3, 96, 192], 0.0034, reuse)
    bw3 = init_b("conv3", [192], reuse)
    conv3 = conv2d(pool2, w3, bw3)
    
    # conv4
    w4 = init_w("conv4", [3, 3, 96, 192], 0.0034, reuse)
    bw4 = init_b("conv4", [192], reuse)
    conv4 = conv2d(conv3, w4, bw4)
    
    # conv5
    w5 = init_w("conv5", [3, 3, 96, 192], 0.0034, reuse)
    bw5 = init_b("conv5", [192], reuse)
    conv5 = conv2d(conv4, parameters['w5'], parameters['bw5'])
    pool5 = max_pool(conv5, 2)
                
    # FC1
    wfc1 = init_w("fc1", [96*32*32, 1024], 1e-2, reuse)
    bfc1 = init_b("fc1", [1024], reuse)
    shape = pool5.get_shape()
    reshape = tf.reshape(pool5, [-1, shape[1].value*shape[2].value*shape[3].value])
    fc1 = tf.nn.relu(tf.matmul(reshape, wfc1) + bfc1)
    fc1_drop = tf.nn.dropout(fc1, keep_prob=DROPOUT)
    
    # FC2
    wfc2 = init_w("fc2", [1024, 1024], 1e-2, reuse)
    bfc2 = init_b("fc2", [1024], reuse)
    fc2 = tf.nn.relu(tf.matmul(fc1_drop, wfc2) + bfc2)
    fc2_drop = tf.nn.dropout(fc2, keep_prob=DROPOUT)
    
    # FC3
    wfc3 = init_w("fc3", [1024, 10], 1e-2, reuse)
    bfc3 = init_b("fc3", [10], reuse)
    softmax_linear = tf.add(tf.matmul(fc2_drop, wfc3), bfc3)
    
    return softmax_linear


def loss_function(logits, labels):
    '''return loss'''
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    tf.add_to_collection('losses', cross_entropy_mean)
    return cross_entropy_mean


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
        
        # validation step
        
        with tf.device('/cpu:0'):
            validation_images, validation_labels = input.get_validation_batch_data(BATCH_SIZE)
        
        validation_logits = inference(validation_images, True)
        validation_labels = tf.one_hot(validation_labels, depth=10)
        
        # 需要把 每一个batch的accuracy加起来，求平均
        correct_pred = tf.equal(tf.argmax(validation_logits, 1), tf.argmax(validation_labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        store_accuracy = tf.add_to_collection('accuracy', accuracy)
        
        add_global = global_step.assign_add(1)
        
        with tf.train.MonitoredTrainingSession(
            hooks=[tf.train.StopAtStepHook(last_step=EPOCH_NUM)]) as mon_sess:
            
            while not mon_sess.should_stop():
                mon_sess.run(train_op)
                step = mon_sess.run(add_global)
                
                if step % 700 == 0:
                    print mon_sess.run(tf.get_collection('losses'))
                    print mon_sess.run(tf.get_collection('learning_rate'))
                    
                    for i in range(78):
                        mon_sess.run(calc_accuracy)
                        
                    accuracy = mon_sess.run(tf.add_n(tf.get_collection("accuracy")))
                    accuracy /= 78
                    print("%d  validation: %f" % (step, accuracy))
                
                
def main():
    train()
  

if __name__ == "__main__":
    main()
    