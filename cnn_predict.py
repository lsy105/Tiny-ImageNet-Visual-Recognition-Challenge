import tensorflow as tf
from read_data import *
import numpy as np
from util import *
from tensorflow.tensorboard.tensorboard import main
from layers import *
from cnn import *


img_W = 56         #image weight
img_H = 56         #image height
img_C = 3          #image channel
F_dim = 3          #3x3
pool_dim = 2       #pool layer dimension


LR = 0.6e-3                                                              #learning rate
reg_rate = 8e-5                                                          #regularization rate
num_class = 200                                                          #number of classes  
keep_rate = 0.8                                                          #keep_rate = 1 - dropout rate 
batch_size = 32                                                          #batch size
num_epoch = 20                                                           #number of epoch want to run
conv_filter_list = [32, 32, 64, 64, 128, 128]                            #conv layer filter channel list 
FC_layer_list = [600, 600, 200]                                          #FC layer size list
model_path = './cnn_model'                                               #path to saved model
test_data_path = '/home/lsy/cs231n/tiny-imagenet-200/val'                #path to test dataset
result_filename = 'cnn_test.txt'                                         #file for saving result

#placeholder layer
X = tf.placeholder(tf.float32, shape=[None, img_H, img_W, img_C])
keep_prob = tf.placeholder(tf.float32)
training = tf.placeholder(tf.bool)

#obtain the score matrix, softmax is not applied yet
output = CNN(X, conv_filter_list, FC_layer_list, keep_prob, training) 
prediction = tf.argmax(output, 1)
saver = tf.train.Saver()
init_op = tf.group(tf.initialize_all_variables())

with tf.Session() as test:
      test.run(init_op)
 
      #load process training images
      test_imgs,filename_list = LoadProcessImages(test_data_path,
                                                   'test')
       
      saver.restore(test, model_path)
      test_imgs = test_imgs[:, 4:60, 4:60, :]
      
      result = list()
      idx = 0
      while idx < test_imgs.shape[0]:
            img = test_imgs[idx:idx + 32]
            sample = test.run(prediction, feed_dict={X: img,
                                                     keep_prob: 1.0, 
                                                     training: 0})
            for element in sample:
                  result.append(element)
            idx += 32
      PrintTestResult(filename_list, result, result_filename) 
