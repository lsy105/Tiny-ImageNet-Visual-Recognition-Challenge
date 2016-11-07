import tensorflow as tf
from util import *

def Conv1x1(x, out_C, training_mode):
      #get dimensions of x
      x_size, x_H, x_W, x_C = x.get_shape()
      x_H, x_W, x_C = int(x_H), int(x_W), int(x_C)
      
      #create weights and bias
      W = InitWeights([1, 1, x_C, out_C], name='AW')
      b = InitBias([out_C], name='b')
      
      #do 1x1 conv 
      conv_out = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
      conv_out = tf.contrib.layers.batch_norm(conv_out, decay=0.99, center=True,
                                              scale=True, epsilon=1e-10,
                                              updates_collections=None,
                                              is_training=training_mode)
      return conv_out



def ConvNormReluForward(x, out_C, training_mode):
      #get dimensions of x 
      x_size, x_H, x_W, x_C = x.get_shape()
      x_H, x_W, x_C = int(x_H), int(x_W), int(x_C)     
    
      #create weight and bias
      W = InitWeights([3, 3, x_C, out_C], name='AW')
      b = InitBias([out_C], name='b')

      #conv batch and relu
      conv_out = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
      conv_out += b
      conv_out = tf.contrib.layers.batch_norm(conv_out, decay=0.99, center=True,
                                              scale=True, epsilon=1e-10,
                                              updates_collections=None,
                                              is_training=training_mode)
      conv_out = tf.nn.relu(conv_out)
      return conv_out


def Conv7x7NormReluForward(x, out_C, training_mode):
      #get dimensions of x 
      x_size, x_H, x_W, x_C = x.get_shape()
      x_H, x_W, x_C = int(x_H), int(x_W), int(x_C)

      #create weight and bias
      W = InitWeights([7, 7, x_C, out_C], name='AW')
      b = InitBias([out_C], name='b')

      #conv batch and relu
      conv_out = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
      conv_out += b
      conv_out = tf.contrib.layers.batch_norm(conv_out, decay=0.99, center=True,
                                              scale=True, epsilon=1e-10,
                                              updates_collections=None,
                                              is_training=training_mode)
      conv_out = tf.nn.relu(conv_out)
      return conv_out


def ConvNormForward(x, out_C, training_mode):
      #get dimensions of x 
      x_size, x_H, x_W, x_C = x.get_shape()
      x_H, x_W, x_C = int(x_H), int(x_W), int(x_C)

      #create weight and bias
      W = InitWeights([3, 3, x_C, out_C], name='AW')
      b = InitBias([out_C], name='b')

      #conv batch and relu
      conv_out = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
      conv_out += b
      conv_out = tf.contrib.layers.batch_norm(conv_out, decay=0.99, center=True,
                                              scale=True, epsilon=1e-10,
                                              updates_collections=None,
                                              is_training=training_mode)
      return conv_out


def FCLayer(x, out_size, keep_prob, is_output_layer=0, training_mode=1):
      x_row, x_col = x.get_shape()
      W = InitWeights([x_col, out_size], name='AW')
      b = InitBias([out_size], name='b')
      layer_out = tf.matmul(x, W) + b
      if not is_output_layer:
            layer_out = tf.contrib.layers.batch_norm(layer_out, decay=0.99,
                                                     center=True, scale=True, 
                                                     epsilon=1e-10,
                                                     updates_collections=None,
                                                     is_training=training_mode)
            layer_out = tf.nn.relu(layer_out)

            #add dropout for training
            if training_mode is not None:
                  layer_out = tf.nn.dropout(layer_out, keep_prob)
      return layer_out
