import os, datetime
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm
from DataLoader import *

# Dataset Parameters
batch_size = 256
load_size = 256
fine_size = 224
c = 3
data_mean = np.asarray([0.45834960097,0.44674252445,0.41352266842])

# Training Parameters
learning_rate = 0.001
dropout = 0.5 # Dropout, probability to keep units
training_iters = 15000
step_display = 50
step_save = 10000
path_save = 'alexnet_bn'
start_from = ''

opt_data_test = {
    'data_root': '../../data/images/',   # MODIFY PATH ACCORDINGLY
    'data_list': '../../data/train.txt',   # MODIFY PATH ACCORDINGLY
    'load_size': load_size,
    'fine_size': fine_size,
    'data_mean': data_mean,
    'randomize': False
    }

loader_test = DataLoaderDisk(**opt_data_test)

# tf Graph input

graph = tf.get_default_graph()
x = graph.get_tensor_by_name("x:0")
y = graph.get_tensor_by_name("y:0")
keep_dropout = graph.get_tensor_by_name("keep_dropout:0")
train_phase = graph.get_tensor_by_name("train_phase:0")

accuracy1 = graph.get_tensor_by_name("accuracy1:0")
accuracy5 = graph.get_tensor_by_name("accuracy5:0")

saver = tf.train.import_meta_graph('../../model_files/alexnet_bn-5800.meta')

with tf.Session() as sess:
    # Initialization
    saver.restore(sess, tf.train.latest_checkpoint('../../model_files'))

    step = 0

    # Evaluate on the whole validation set
    print('Evaluation on the whole validation set...')
    num_batch = loader_val.size()//batch_size
    acc1_total = 0.
    acc5_total = 0.
    loader_test.reset()
    for i in range(num_batch):
        images_batch, labels_batch = loader_test.next_batch(batch_size)
        acc1, acc5 = sess.run([accuracy1, accuracy5], feed_dict={x: images_batch, y: labels_batch, keep_dropout: 1., train_phase: False})
        acc1_total += acc1
        acc5_total += acc5
        print("Test Accuracy Top1 = " + \
              "{:.4f}".format(acc1) + ", Top5 = " + \
              "{:.4f}".format(acc5))

    acc1_total /= num_batch
    acc5_total /= num_batch
    print('Evaluation Finished! Accuracy Top1 = ' + "{:.4f}".format(acc1_total) + ", Top5 = " + "{:.4f}".format(acc5_total))
