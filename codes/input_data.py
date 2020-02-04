#By @Yao lee based on Kevin Xu's coding
#liyaojc2@163.com
#
#The aim of this project is to use TensorFlow to fusing our muliti-data.
#    - input_data.py:  read in data and generate batches
#    - model: build the model architecture
#    - training: train

# I used Ubuntu with Python 2.7, TensorFlow 1.0*, other OS should also be good.
# With current settings, 2000 traing steps .


# data: from public remote image

# How to run?
# 1. run the classifyUrl.py once
# 2. call the classificater().

# Note:
#you should change the logs_train_dir to your own dir in classifyUrl.py
#change the multi-data-dir on your own dir
#in input_data have different coding depended on your intension read careful


import tensorflow as tf
import numpy as np
import os
import random


# you need to change this to your data directory
randomList = []
def get_files(file_dir, total):
    '''
    Args:
        file_dir: file directory
    Returns:
        list of images and labels
    '''
    cats = []
    label_cats = []
    dogs = []
    label_dogs = []
    for file in os.listdir(file_dir):
        name = file.split('.')
        if name[0]=='cat':
            cats.append(file_dir + file)
            label_cats.append(0)
        else:
            dogs.append(file_dir + file)
            label_dogs.append(1)
    print('There are %d baseballDiamond\nThere are %d golfCourse' %(len(cats), len(dogs)))
    image_list = []
    label_list = []
    image = np.hstack((cats, dogs))
    label = np.hstack((label_cats, label_dogs))
    #random select data
    randomList = rando(total)

    # the number in range depended on your number of dataset and Also corresponds to the line 79
    for i in range(0, total):
        order = randomList[i]
        image_list.append(image[order])

        label_list.append(label[order])
    temp = np.array([image_list, label_list])
    temp = temp.transpose()

    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [float(i) for i in label_list]
    return image_list, label_list,randomList


def rando(total):
    value = range(total)
    randomList=np.random.permutation(value)
    return randomList


def get_batch(image, label, image_W, image_H, batch_size, capacity):
    '''
    Args:
        image: list type
        label: list type
        image_W: image width
        image_H: image height
        batch_size: batch size
        capacity: the maximum elements in queue
    Returns:
        image_batch: 4D tensor [batch_size, width, height, 3], dtype=tf.float32
        label_batch: 1D tensor [batch_size], dtype=tf.int32
    '''
    
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)

    # make an input queue
    input_queue = tf.train.slice_input_producer([image, label])
    
    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents, channels=3)
    
    ######################################
    # data argumentation should go to here
    ######################################
    
    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)
    
    # if you want to test the generated batches of images, you might want to comment the following line.
    image = tf.image.per_image_standardization(image)
    
    image_batch, label_batch = tf.train.batch([image, label],
                                                batch_size= batch_size,
                                                num_threads= 64,
                                                capacity = capacity)
    
    #you can also use shuffle_batch 
#    image_batch, label_batch = tf.train.shuffle_batch([image,label],
#                                                      batch_size=BATCH_SIZE,
#                                                      num_threads=64,
#                                                      capacity=CAPACITY,
#                                                      min_after_dequeue=CAPACITY-1)
    
    label_batch = tf.reshape(label_batch, [batch_size])
    image_batch = tf.cast(image_batch, tf.float32)
    
    return image_batch, label_batch

