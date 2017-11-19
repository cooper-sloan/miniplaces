import os, datetime
import pickle
import numpy as np
import tensorflow as tf
import alexnet_model
import resnet_model
from tensorflow.contrib.layers.python.layers import batch_norm
from DataLoader import *
import json

# Dataset Parameters
batch_size = 256
load_size = 256
fine_size = 224
c = 3
data_mean = np.asarray([0.45834960097,0.44674252445,0.41352266842])

# Training Parameters
learning_rate = 0.0001
dropout = 0.5 # Dropout, probability to keep units
training_iters = 5000
step_display = 50
step_save = 200
path_save = './model_out/res'
# start_from = './model_out/res-1000'
start_from=''

def batch_norm_layer(x, train_phase, scope_bn):
    return batch_norm(x, decay=0.9, center=True, scale=True,
            updates_collections=None,
            is_training=train_phase,
            reuse=None,
            trainable=True,
            scope=scope_bn)


    # Construct dataloader
opt_data_train = {
        #'data_h5': 'miniplaces_256_train.h5',
        'data_root': '../data/images/',   # MODIFY PATH ACCORDINGLY
        'data_list': '../data/train.txt', # MODIFY PATH ACCORDINGLY
        'load_size': load_size,
        'fine_size': fine_size,
        'data_mean': data_mean,
        'randomize': True,
        'perm': True
        }
opt_data_val = {
        #'data_h5': 'miniplaces_256_val.h5',
        'data_root': '../data/images/',   # MODIFY PATH ACCORDINGLY
        'data_list': '../data/val.txt',   # MODIFY PATH ACCORDINGLY
        'load_size': load_size,
        'fine_size': fine_size,
        'data_mean': data_mean,
        'randomize': False,
        'perm': True
        }

loader_train = DataLoaderDisk(**opt_data_train)
loader_val = DataLoaderDisk(**opt_data_val)
#loader_train = DataLoaderH5(**opt_data_train)
#loader_val = DataLoaderH5(**opt_data_val)

x = tf.placeholder(tf.float32, [None, fine_size, fine_size, c],name="x")
y = tf.placeholder(tf.int64, None,name="y")
keep_dropout = tf.placeholder(tf.float32,name="keep_dropout")
train_phase = tf.placeholder(tf.bool,name="train_phase")

# logits = tf.identity(alexnet_model.alexnet(x, keep_dropout, train_phase),name='logits')
res = resnet_model.imagenet_resnet_v2(18, 100)
logits = tf.identity(res(x,train_phase),name='logits')

loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits))
train_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

accuracy1 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits, y, 1), tf.float32),name="accuracy1")
accuracy5 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits, y, 5), tf.float32),name="accuracy5")

init = tf.global_variables_initializer()

saver = tf.train.Saver()

#writer = tf.train.SummaryWriter('.', graph=tf.get_default_graph())

convergence_data = {
        'name': 'conergence_data1',
        'top1': [],
        'top5': [],
        'learning_rate': learning_rate,
        'dropout': dropout,
        'notes': '',
        }
# Launch the graph
with tf.Session() as sess:
    if len(start_from)>1:
        saver.restore(sess, start_from)
    else:
        sess.run(init)

    step = 0

    while step < training_iters:
        images_batch, labels_batch = loader_train.next_batch(batch_size)

        if step % step_display == 0:
            print('[%s]:' %(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

            l, acc1, acc5 = sess.run([loss, accuracy1, accuracy5], feed_dict={x: images_batch, y: labels_batch, keep_dropout: 1., train_phase: True})
            print("-Iter " + str(step) + ", Training Loss= " + \
                    "{:.6f}".format(l) + ", Accuracy Top1 = " + \
                    "{:.4f}".format(acc1) + ", Top5 = " + \
                    "{:.4f}".format(acc5))

            images_batch_val, labels_batch_val = loader_val.next_batch(batch_size)
            l, acc1, acc5 = sess.run([loss, accuracy1, accuracy5], feed_dict={x: images_batch_val, y: labels_batch_val, keep_dropout: 1., train_phase: True})
            print("-Iter " + str(step) + ", Validation Loss= " + \
                    "{:.6f}".format(l) + ", Accuracy Top1 = " + \
                    "{:.4f}".format(acc1) + ", Top5 = " + \
                    "{:.4f}".format(acc5))

            convergence_data['top1'].append(acc1)
            convergence_data['top5'].append(acc5)
        sess.run(train_optimizer, feed_dict={x: images_batch, y: labels_batch, keep_dropout: dropout, train_phase: True})

        step += 1

        if step % step_save == 0:
            saver.save(sess, path_save, global_step=step)
            print("Model saved at Iter %d !" %(step))
            with open(convergence_data['name']+'.txt', 'w') as f:
                pickle.dump(convergence_data,f)


    print("Optimization Finished!")

    # Evaluate on the whole validation set
    print('Evaluation on the whole validation set...')
    num_batch = loader_val.size()//batch_size
    acc1_total = 0.
    acc5_total = 0.
    loader_val.reset()
    for i in range(num_batch):
        images_batch, labels_batch = loader_val.next_batch(batch_size)
        acc1, acc5 = sess.run([accuracy1, accuracy5], feed_dict={x: images_batch, y: labels_batch, keep_dropout: 1., train_phase: False})
        acc1_total += acc1
        acc5_total += acc5
        print("Validation Accuracy Top1 = " + \
                "{:.4f}".format(acc1) + ", Top5 = " + \
                "{:.4f}".format(acc5))

        acc1_total /= num_batch
    acc5_total /= num_batch
    print('Evaluation Finished! Accuracy Top1 = ' + "{:.4f}".format(acc1_total) + ", Top5 = " + "{:.4f}".format(acc5_total))
