from util import *
from read_data import *
from layers import *
import tensorflow as tf


def Res3x3(x, out_C, is_pooling, training_mode):
      """
      create a residual net with 2 layers
  
      Args:
            x: input of residual net
            out_C: number of output channels for this single residual net
            is_pooling: if do pooling in the begining of this net 
                        (2x2 pool with stride 2)

      Returns:
            conv_out: a tensor with shape [N, x_W, x_H, out_C] x_W is 
            width of x, x_H is height of x
      """

      if is_pooling:
            x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], 
                               strides=[1, 2, 2, 1], 
                               padding='SAME')
  
      x_size, x_H, x_W, x_C = x.get_shape()
      x_H, x_W, x_C = int(x_H), int(x_W), int(x_C)

      #first conv layer
      conv_out = ConvNormReluForward(x, out_C, training_mode)
 
      #add dropout
      if training_mode is not None:
            conv_out = tf.nn.dropout(conv_out, 0.8)           
                              
      #second conv layer without relu
      conv_out = ConvNormForward(conv_out, out_C, training_mode)

      #shortcut path
      if x_C != out_C:
            x = Conv1x1(x, out_C, training_mode)
      conv_out += x
      conv_out = tf.nn.relu(conv_out)
      return conv_out                        
                          
                                                      
      
      
def Resnet(x, num_class, training_mode):
      """
      build 20 layers residual neural network
      Args:
            x: input with shape [batch_size, width, height, channel]
            training_mode: if in training mode

      Returns:
            return the score matrix with shape [batch_size, num_class(200)] 
      """
      #define number of output channel of filters in each resdidual unit
      filter_list  = [64, 64, 128, 128, 256, 256, 256]                         
      #define if do pooling at current residual unit(1: do pooling)
      if_pool_list = [ 1,  0,  1,  0,  1,  0,  0]                
      num_pool = sum(if_pool_list)
     
      #get shape of x
      x_size, x_W, x_H, x_C = x.get_shape()     
      x_H, x_W, x_C = int(x_H), int(x_W), int(x_C)

      #verify if there are too many pooling layer
      assert x_W % (2**num_pool) == 0, "too many pooling layer"
 
      #first conv layer 3x3 filter
      with tf.name_scope('first_layer'):
            conv_out = ConvNormReluForward(x, filter_list[0], training_mode)
            conv_out = ConvNormReluForward(conv_out, filter_list[0], training_mode)
      
      #build all residual units
      with tf.name_scope('residual_network'):  
            for idx in range(len(filter_list)):
                  with tf.name_scope('unit_%d_%d' % (idx, filter_list[idx])):
                        conv_out = Res3x3(conv_out, filter_list[idx], 
                                          if_pool_list[idx], training_mode)             
      
      #last layer
      with tf.name_scope('output_layer'):
            last_layer_size = filter_list[-1] * (x_W // (2**num_pool))**2 
            conv_out = tf.reshape(conv_out, [-1, last_layer_size])
            output = FCLayer(conv_out, num_class, 1.0, 1, training_mode)
      return output


