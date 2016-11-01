import tensorflow as tf
from read_data import *

min_capacity = 10000 
def InitWeights(shape, name=''):
      """  
      initilize weights with zero mean and std provided by user
      calibrate the variance with sqrt(2/n) so that output has the same
      variance as input. n is the fan-in of the weight.
            
      Args:
            shape: shape of the weight matrix
                   weight of convolution layer [F_dim, F_dim, 
                                                num_input_feature_map, 
                                                num_output_feature_map]
            name:  name of this variable

      Returns:
            weight_matrix: tensorflow variables with the same shape as user specified
                           and initial values with zero mean and std specified by user 
      """
      #calculate fan-in and fan_out of weight matrix
      shape = list(map(int, shape))
      fan_in = 0
      fan_out = 0
      if(len(shape) == 2):
            fan_in = shape[0]
            fan_out = shape[1]
      else:
            fan_in  = shape[0] * shape[1] * shape[2]
            fan_out = shape[0] * shape[1] * shape[3]
      weight_matrix = tf.truncated_normal(shape, mean=0.0, stddev=tf.sqrt(2.0/(fan_in + fan_out)))
      return tf.Variable(weight_matrix, name=name)



def InitBias(shape, name=''):
      """  
      initilize bias with 0.0
      
      Args:
            shape: shape of the bias matrix
            name:  name of this bias 
 
      Returns:
            bias_matrix: tensorflow variables with the same shape as user specified
                         and initial values with zero mean and std specified by user 
      """

      bias_matrix = tf.constant(0.0, shape = shape)
      return tf.Variable(bias_matrix, name=name)



def Loss(score_matrix, bool_matrix, reg_rate):
      """
      calculate loss
      
      Args:
            score_matrix: score_matrix with shape[batch_size, num_class]
            bool_matrix:  matrix with shape[batch_size, num_class]
                          most elements are zeros except the correct class element 
            reg_rate:     regulation rate for L2 loss 
      Returns:
            loss: the total loss 
      """
      output = tf.nn.softmax(score_matrix)
      #calculate loss with L2
      loss = tf.reduce_mean(-tf.reduce_sum(bool_matrix * tf.log(output), reduction_indices=[1]))

      #calculate weight L2 loss
      W_loss = 0.0
      for W in tf.trainable_variables():
            if W.op.name.find(r'AW') > 0:
                  print (W.op.name)
                  W_loss += tf.nn.l2_loss(W)      
      loss += reg_rate * W_loss
      print (reg_rate * W_loss)
      return loss


def MeanSquareLoss(score_matrix, truth_matrix, reg_rate):
      """
      calculate loss
      
      Args:
            score_matrix: score_matrix with shape[batch_size, num_class]
            bool_matrix:  matrix with shape[batch_size, num_class]
                          most elements are zeros except the correct class element 
            reg_rate:     regulation rate for L2 loss 
      Returns:
            loss: the total loss 
      """
      #calculate loss with L2
      loss = tf.reduce_mean(tf.square(score_matrix - truth_matrix))

      #calculate weight L2 loss
      W_loss = 0.0
      for W in tf.trainable_variables():
            if W.op.name.find(r'AW') > 0:
                  print (W.op.name)
                  W_loss += tf.nn.l2_loss(W)      
      loss += reg_rate * W_loss
      print (reg_rate * W_loss)
      return loss


def Accuracy(score_matrix, y_t):
      """
      calculate accuracy for batch
      
      Args:
            score_matrix: score_matrix
            y_t   : truth matrix

      Returns:
            accuracy: accuracy rate
      """
      correct_prediction = tf.equal(tf.argmax(score_matrix,1), tf.argmax(y_t,1))
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
      return accuracy


def RandomCrop(image):
      """ 
      radnom crop a image with size 56x56
      """
      row = np.random.randint(9)
      col = np.random.randint(9)
      new_image = image[row:row + 56, col:col + 56, :]
      return new_image
      


def GenBatch(img_in, labels, batch_size=32, num_class=200, random_crop=1):
      """
      Generate a batch
      Args:
            img_in: image dataset with shape[N, W, H, C]. N is number of images, 
                    W is width and H is height and C is channel
            labels: labels for images with shape [N, 1]
            batch_size: size of batch
      Returns:
            img_out: a np.array of image data with shape[batch_size, img_width, 
                                                         img_height, img_channel]
            label_matrix: label np.array with shape[batch_size, num_class]
      """
      #create a label matrix with shape [batch_size, num_class]
      if not random_crop:
            label_matrix = np.zeros((batch_size, num_class))
            idx = np.random.choice(labels.shape[0], batch_size)
            label_matrix[np.arange(batch_size), labels[idx]] = 1.0 
            #crop from the center of image with shape 56x56
            img_out = img_in[idx, 4:60, 4:60, :]
      else:
            batch_size = batch_size // 2
            label_matrix = np.zeros((batch_size, num_class))
            idx = np.random.choice(labels.shape[0], batch_size)
            label_matrix[np.arange(batch_size), labels[idx]] = 1.0 
            img_out = img_in[idx]
            #create cropped images
            temp_array = np.empty((batch_size, 56, 56, 3))
            temp_label = np.empty((batch_size, num_class))
            for i in range(batch_size):
                  new_image = RandomCrop(img_out[i])
                  temp_array[i] = new_image
                  temp_label[i] = np.copy(label_matrix[i])
            img_out = img_out[:, 4:60, 4:60, :]
            img_out = np.concatenate((img_out, temp_array), axis=0) 
            label_matrix = np.concatenate((label_matrix, temp_label), axis=0)
      return img_out, label_matrix


def Train(LR, loss):
      """
      training the model(use adam algorithm)

      Args:
            LR: learning rate
            loss: scalar loss of this model
      Returns:
            train_step: training operation
      """
      train_step = tf.train.AdamOptimizer(learning_rate=LR).minimize(loss)
      return train_step

def PrintTestResult(file_list, result, output_file):
      """
      print test result with the format "image_name class"
     
      Args:
            result: list of predicted labels for each test image.
            output_filename: path of output file
      """ 
      map_list = list()
      with open("./train_label") as fin:
            for line in fin:
                  list_t = line.split(" ")
                  map_list.append(list_t[0])
      fin.close()

      with open(output_file, 'w') as fout:
            for idx in range(len(result)):
                  line = file_list[idx] + " " + map_list[result[idx]]
                  fout.write(line + '\n')
      fout.close()
      
                                          
