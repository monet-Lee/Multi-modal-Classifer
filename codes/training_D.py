import os
import numpy as np
import tensorflow as tf
import input_dataD
import modelD
from PIL import Image
import matplotlib.pyplot as plt


def run_training(N_CLASSES, IMG_W, IMG_H, BATCH_SIZE, MAX_STEP, CAPACITY,
                 model2_data, learning_rate, randomList, total):
 
    train, train_label = input_dataD.get_files(model2_data, randomList, total)
    train_batch, train_label_batch = input_dataD.get_batch(train,
                                                          train_label,
                                                          IMG_W,
                                                          IMG_H,
                                                          BATCH_SIZE,
                                                          CAPACITY)
    train_logits = modelD.inference(train_batch, BATCH_SIZE, N_CLASSES)
    return train_logits


def get_one_image(train,i):
    '''Randomly pick one image from training data
    Return: ndarray
    '''
    img_dir = train[i]

    image = Image.open(img_dir)
    plt.imshow(image)
    image = image.resize([32, 32])
    image = np.array(image)
    return image


def evaluate_D_image(test2_data, randomList, i, total, N_CLASSES):
    '''Test one image against the saved models and parameters
    '''
    train, train_label = input_dataD.get_files(test2_data, randomList, total)
    image_array = get_one_image(train,i)
    image = tf.cast(image_array, tf.float32)
    image = tf.image.per_image_standardization(image)
    image = tf.reshape(image, [1, 32, 32, 3])
    logit = modelD.inference(image, 1, N_CLASSES)
    return logit
#
