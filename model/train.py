import os, datetime
import pickle
import numpy as np
import tensorflow as tf
import alexnet_model
import resnet_model
from tensorflow.contrib.layers.python.layers import batch_norm
from DataLoader import *
import json
import time

# Dataset Parameters
batch_size = 256
load_size = 256
fine_size = 224
c = 3
data_mean = np.asarray([0.45834960097,0.44674252445,0.41352266842])
run_evaluation = False

# Training Parameters
learning_rate = 0.0001
dropout = 0.5 # Dropout, probability to keep units
training_iters = 1
step_display = 50
step_save = 200
path_save = './model_out/res'
start_from = './model_out/res-5000'
# start_from=''

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
opt_data_test = {
    'data_root': '../data/images/',   # MODIFY PATH ACCORDINGLY
    'data_list': '../data/test.txt',   # MODIFY PATH ACCORDINGLY
    'load_size': load_size,
    'fine_size': fine_size,
    'data_mean': data_mean,
    'randomize': False,
    'perm': False
}

loader_train = DataLoaderDisk(**opt_data_train)
loader_val = DataLoaderDisk(**opt_data_val)
loader_test = DataLoaderDisk(**opt_data_test)
#loader_train = DataLoaderH5(**opt_data_train)
#loader_val = DataLoaderH5(**opt_data_val)

x = tf.placeholder(tf.float32, [None, fine_size, fine_size, c],name="x")
y = tf.placeholder(tf.int64, None,name="y")
keep_dropout = tf.placeholder(tf.float32,name="keep_dropout")
train_phase = tf.placeholder(tf.bool,name="train_phase")

# logits = tf.identity(alexnet_model.alexnet(x, keep_dropout, train_phase),name='logits')
res = resnet_model.imagenet_resnet_v2(18, 100)
logits = res(x,train_phase)
values,indices = tf.nn.top_k(logits,k=5)

loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits))
train_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

accuracy1 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits, y, 1), tf.float32),name="accuracy1")
accuracy5 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits, y, 5), tf.float32),name="accuracy5")

init = tf.global_variables_initializer()

saver = tf.train.Saver()

#writer = tf.train.SummaryWriter('.', graph=tf.get_default_graph())

convergence_data = {
        'name': 'conergence_data1',
        'top1_t': [],
        'top5_t': [],
        'top1_v': [],
        'top5_v': [],
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

            convergence_data['top1_t'].append(acc1)
            convergence_data['top5_t'].append(acc5)

            images_batch_val, labels_batch_val = loader_val.next_batch(batch_size)
            l, acc1, acc5 = sess.run([loss, accuracy1, accuracy5], feed_dict={x: images_batch_val, y: labels_batch_val, keep_dropout: 1., train_phase: True})
            print("-Iter " + str(step) + ", Validation Loss= " + \
                    "{:.6f}".format(l) + ", Accuracy Top1 = " + \
                    "{:.4f}".format(acc1) + ", Top5 = " + \
                    "{:.4f}".format(acc5))

            convergence_data['top1_v'].append(acc1)
            convergence_data['top5_v'].append(acc5)
        sess.run(train_optimizer, feed_dict={x: images_batch, y: labels_batch, keep_dropout: dropout, train_phase: True})

        step += 1

        if step % step_save == 0:
            saver.save(sess, path_save, global_step=step)
            print("Model saved at Iter %d !" %(step))
            with open(convergence_data['name']+'.txt', 'w') as f:
                pickle.dump(convergence_data,f)


    print("Optimization Finished!")

    # Evaluate on the whole validation set
    if run_evaluation:
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

    num_batch = loader_test.size()//batch_size
    print('Running inference on test data')
    with open("../data/test.txt",'r') as f:
        with open("./predictions/prediction_"+str(time.time())+".txt",'w') as fo:
            loader_test.reset()
            out_lines = []
            in_lines = f.readlines()
            i = 0
            for j in range(num_batch):
                images_batch, labels_batch = loader_test.next_batch(batch_size)
                top_ks = sess.run(indices, feed_dict={x: images_batch, keep_dropout: 1., train_phase: False})
                for top_k in top_ks:
                    in_line = in_lines[i].split(" ")
                    i += 1
                    file_name = in_line[0]
                    out_line = file_name
                    for v in top_k:
                        out_line += " " + str(v)
                    out_lines.append(out_line)
            fo.write("\n".join(out_lines))
