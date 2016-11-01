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
train_data_path = '/home/lsy/cs231n/new_data/tiny-imagenet-200/train'    #path to train dataset
val_data_path = '/home/lsy/cs231n/tiny-imagenet-200/val'                 #path to val dataset

#placeholder layer
X   = tf.placeholder(tf.float32, shape=[None, img_H, img_W, img_C])
y_t = tf.placeholder(tf.float32, shape=[None, num_class])
keep_prob = tf.placeholder(tf.float32)
training = tf.placeholder(tf.bool)

#obtain the score matrix, softmax is not applied yet
output = CNN(X, conv_filter_list, FC_layer_list, keep_prob, training) 
loss = Loss(output, y_t, reg_rate)
acc = Accuracy(output, y_t)
train_step = Train(LR, loss)

#add summary
loss_summary = tf.scalar_summary('loss', loss)
train_acc_summary  = tf.scalar_summary('training accuracy', acc)
val_acc_summary    = tf.scalar_summary('val accuracy', acc)

saver = tf.train.Saver()
init_op = tf.group(tf.initialize_all_variables())

with tf.Session() as test:
      test.run(init_op)

      #save tensorboard data in ./train directory
      train_writer = tf.train.SummaryWriter('./train', test.graph)
 
      #load process training images
      train_imgs, train_labels = LoadProcessImages(train_data_path, 'training')

      #load process val images
      val_imgs, val_labels = LoadProcessImages(val_data_path, 'val')

      data_size = train_labels.shape[0]
      print_iter = 10
      num_iter = int(num_epoch * data_size / batch_size) 
      for i in range(num_iter):
            img_batch, label_batch = GenBatch(train_imgs, train_labels, 
                                              batch_size, num_class)       
            run_loss, summary = test.run([loss, loss_summary], feed_dict={X: img_batch, 
                                                                          y_t: label_batch, 
                                                                          keep_prob: keep_rate, 
                                                                          training: 1})
            #loss summary
            train_writer.add_summary(summary, i)

            train_step.run(feed_dict={X: img_batch, y_t: label_batch, 
                                      keep_prob: keep_rate, training: 1})

            if i % print_iter == 0:
                  if i == 0: print (run_loss)
                  
                  #training acc summary
                  train_summary = test.run(train_acc_summary, feed_dict={X: img_batch, 
                                                                         y_t: label_batch, 
                                                                         keep_prob: 1.0, 
                                                                         training: 0})
                  train_writer.add_summary(train_summary, i)

                  #val acc summary
                  img_batch, label_batch = GenBatch(val_imgs, val_labels, 
                                                    batch_size, num_class, random_crop=0)       

                  val_summary = test.run(val_acc_summary, feed_dict={X: img_batch, 
                                                                     y_t: label_batch, 
                                                                     keep_prob: 1.0, 
                                                                     training: 0})
                  train_writer.add_summary(val_summary, i)
            i += 1
            if i % 1000 == 0 and i > 0: saver.save(test, "./model")
