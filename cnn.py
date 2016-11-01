from util import *
import tensorflow as tf
from layers import *


def ConvPattern(x, out_C, training_mode=1):
      #[CONV -> RELU -> CONV -> RELU -> POOL] 
      #first conv, batch_norm and relu
      h_conv1 = ConvNormReluForward(x, out_C, training_mode)

      #second conv and relu 
      h_conv2 = ConvNormReluForward(h_conv1, out_C, training_mode)
     
      #2x2 max pool
      conv_out = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
      return conv_out



      
def CNN(x, conv_filter_list, FC_layer_list, keep_prob, training_mode=1):
      """
      build a CNN network
      Args:
            x: input with shape [batch_size, width, height, channel]
            training_mode: if in training mode

      Returns:
            return the score matrix with shape [batch_size, num_class(200)] 
      """
      num_filter = len(conv_filter_list)
      assert num_filter % 2 == 0, "use even number of filter"
      num_layer = num_filter / 2
      #get shape of x
      x_size, x_H, x_W, x_C = x.get_shape()     

      #verify if there are too many pooling layer
      assert x_W % (2**num_layer) == 0, "too many pooling layer"
 
      #build conv layers
      for idx in range(1, int(num_layer + 1)):
            filter_idx = (idx - 1) * 2  
            with tf.name_scope('conv_pattern_%d' % idx):
                  x = ConvPattern(x, conv_filter_list[filter_idx], 
                                  training_mode)             
      
      #FC layers
      N, H, W, C = x.get_shape() 
      output = tf.reshape(x, [-1, int(H * W * C)])

      for idx in range(1, int(num_layer + 1)):
            if idx == num_layer: is_output_layer = 1
            else: is_output_layer = 0
            with tf.name_scope('FC_layer_%d' % idx):
                  output = FCLayer(output, FC_layer_list[idx - 1], 
                                   keep_prob, 
                                   is_output_layer, 
                                   training_mode)
                  
      return output


