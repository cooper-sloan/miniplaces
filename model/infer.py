import os, datetime
import time
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm
from DataLoader import *

batch_size = 1
load_size = 256
fine_size = 224
data_mean = np.asarray([0.45834960097,0.44674252445,0.41352266842])
opt_data_test = {
    'data_root': '../data/images/',   # MODIFY PATH ACCORDINGLY
    'data_list': '../data/test.txt',   # MODIFY PATH ACCORDINGLY
    'load_size': load_size,
    'fine_size': fine_size,
    'data_mean': data_mean,
    'randomize': False,
    'perm': False
}

loader_test = DataLoaderDisk(**opt_data_test)
saver = tf.train.import_meta_graph('./model_out/model-16000.meta')
sess = tf.Session()
saver.restore(sess,'./model_out/model-16000')

graph = tf.get_default_graph()
x = graph.get_tensor_by_name("x:0")
y = graph.get_tensor_by_name("y:0")
keep_dropout = graph.get_tensor_by_name("keep_dropout:0")
train_phase = graph.get_tensor_by_name("train_phase:0")
logits = graph.get_tensor_by_name("logits:0")
softmax = tf.nn.softmax(logits)
values,indices = tf.nn.top_k(softmax,k=5)
print('Inference on the whole test set...')
num_batch = loader_test.size()//batch_size
with open("../data/test.txt",'r') as f:
    with open("./predictions/prediction_"+str(time.time())+".txt",'w') as fo:
        loader_test.reset()
        out_lines = []
        in_lines = f.readlines()
        for i in range(num_batch):
            images_batch, labels_batch = loader_test.next_batch(batch_size)
            p = sess.run(indices, feed_dict={x: images_batch, y: labels_batch, keep_dropout: 1., train_phase: False})
            # p = sess.run(softmax, feed_dict={x: images_batch, y: labels_batch, train_phase: False})
            in_line = in_lines[i].split(" ")
            file_name = in_line[0]
            out_line = file_name
            for v in p[0]:
                out_line += " " + str(v)
            out_lines.append(out_line)
        fo.write("\n".join(out_lines))
