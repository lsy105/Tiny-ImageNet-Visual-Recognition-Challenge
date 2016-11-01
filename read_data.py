import os
import tensorflow as tf
import numpy as np
from scipy import misc

def ClassMapping(map_file):
      """
      Create a dict for mapping the class name nxxxxxxx to int from 1 to 200
      the mapping is created using shell script. the purpose of this function is 
      to read and save the maping in dict
     
      Args:
            map_file: map file path

      return:
            class_map: a dict. key is class name(nXXXXXXXX) and value is the 
            corresponding number            
      """
      class_map = dict()
      with open(map_file) as fin:
            for line in fin:
                  list_t = line.split(" ")
                  class_map[list_t[0]] = int(list_t[1])
      return class_map
            

def ReadTrainImages(input_dir):
      """
      read all images from training dataset
      
      Args:
            dir: directory to the dataset which includes all classes.
            For example, "/tiny-imagenet-200/train/"

      Returns:
            data_list:  np.array of image data [num_images, height, width, depth] 
            label_list: np.array of labels 
      """
      data_list  = list()
      label_list = list()
      class_map = ClassMapping("./train_label")
      for key in class_map.keys():
            path = input_dir + "/" + key + "/images"
            for fi in os.listdir(path):
                  img = misc.imread(path + "/" + fi, mode='RGB')
                  data_list.append(img)
                  label_list.append(class_map[key])
      return (np.array(data_list, dtype=np.float32),  
              np.array(label_list, dtype=np.int32))



def ReadValImages(input_dir):
      """
      read all images from val dataset
      
      Args:
            dir: directory to the val dataset
            For example, "/tiny-imagenet-200/val/"

      Returns:
            data_list:  np.array of image data [num_images, height, width, depth] 
            label_list: np.array of labels

      """
      data_list = list()
      label_list = list()
      class_map = ClassMapping("./train_label")
      label_path = input_dir + "/val_annotations.txt"
      img_path   = input_dir + "/images/"
      with open(label_path, "r") as fin:
            for line in fin:
                  list_t = line.split("\t")
                  img = misc.imread(img_path + list_t[0], mode='RGB')
                  data_list.append(img)
                  label_list.append(class_map[list_t[1]])
      fin.close() 
      return (np.array(data_list, dtype=np.float32), 
              np.array(label_list, dtype=np.int32))


def ReadTestImages(input_dir):
      """
      read all images from val dataset
      
      Args:
            dir: directory to the val dataset
            For example, "/tiny-imagenet-200/test/"

      Returns:
            data_list:  np.array of image data [num_images, height, width, depth] 
            label_list: np.array of file name for each image
      """
      data_list = list()
      file_list = list()
      class_map = ClassMapping("./train_label")
      img_path = input_dir + "/images"
      for fi in os.listdir(img_path):
                  img = misc.imread(img_path + "/" + fi, mode='RGB')
                  data_list.append(img)
                  file_list.append(fi)
      return (np.array(data_list, dtype=np.float32),
              np.array(file_list, dtype=np.str))

 
def DatasetMeanNorm(dataset):
      """
      subtract mean and do normalization for each image
      Args:
            dataset: dataset with shape [N, H, W, C]
                     N data points, Height H, Width W, Channel C

      Return:
            dataset: processed dataset
      """
      N, H, W, C = dataset.shape
      X = dataset.reshape((N, -1))
      X -= np.mean(X, axis = 0)
      X /= np.std(X, axis = 0)
      X = X.reshape((N, H, W, C))
      return X


def DatasetPCAWhitening(dataset):
      N, H, W, C = dataset.shape
      X = dataset.reshape((N, -1))
      X -= np.mean(X, axis = 0)
      cov = np.dot(X.T, X) / N
      U,S,V = np.linalg.svd(cov)
      Xrot = np.dot(X, U)
      Xwhite = Xrot / np.sqrt(S + 1e-10)
      Xwhite = Xwhilte.reshape((N, H, W, C))
      return Xwhite     


def LoadProcessImages(path, data_type='training'):
      """
      Load images and preprocess images with zero mean and normalize variance
      Args:
            path: path to the dataset
            data_type: specify it is test, validatin, or test dataset

      Returns;
            img_array: np.array of all images with shape [size, image]
            label_array: np.array of labels. for test dataset, label_array 
                         is the file name for each images
      """
      if data_type == 'training':
            img_array, label_array = ReadTrainImages(path)
            img_array = DatasetMeanNorm(img_array)
      if data_type == 'val':
            img_array, label_array = ReadValImages(path)
            img_array = DatasetMeanNorm(img_array)
      if data_type == 'test':
            img_array, label_array = ReadTestImages(path)
            img_array = DatasetMeanNorm(img_array)
      return img_array, label_array

