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
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import sys
sys.path.append('codes')
import model
import modelD
import training
import  training_D
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def classifier(N_CLASSES, IMG_W, IMG_H, BATCH_SIZE, MAX_STEP, CAPACITY,
               model1_data, model2_data, learning_rate, logs_dir, total):
    H_Feature, train_batch, train_label_batch, randomList = training.run_training(
        N_CLASSES, IMG_W, IMG_H, BATCH_SIZE, MAX_STEP, CAPACITY, model1_data, learning_rate, total)
    L_Feature = training_D.run_training(N_CLASSES, IMG_W, IMG_H, BATCH_SIZE, MAX_STEP,
         CAPACITY, model2_data, learning_rate, randomList, total)

    # fusion algorithm2
    H_Feature = tf.reshape(H_Feature, [-1])
    L_Feature = tf.reshape(L_Feature, [-1])
    local4 = L_Feature+H_Feature;
    # local4 = tf.reshape(local, shape=[1,-1])
   

    # fusion algorithem 1
    # local4=tf.concat([H_Feature,L_Feature],1)
    with tf.variable_scope('local3', reuse=None) as scope:
        reshape = tf.reshape(local4, shape=[16, -1])
        dim = reshape.get_shape()[1].value
        weights = tf.get_variable('weights',
                                  shape=[dim, 128],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[128],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)

 
    with tf.variable_scope('softmax_linear') as scope:
        weights = tf.get_variable('softmax_linear',
                                  # shape=[256, N_CLASSES],
                                  shape=[128, N_CLASSES],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[N_CLASSES],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        train_logits = tf.add(tf.matmul(local3, weights), biases, name='softmax_linear')


    train_loss = model.losses(train_logits, train_label_batch)
    train_op = model.trainning(train_loss, learning_rate)
    train__acc = model.evaluation(train_logits, train_label_batch)

    summary_op = tf.summary.merge_all()
    sess = tf.Session()
    train_writer = tf.summary.FileWriter(logs_dir, sess.graph)
    saver = tf.train.Saver()

    # transfer learning
    # ckpt = tf.train.get_checkpoint_state(logs_train_dirnew)
    # if ckpt and ckpt.model_checkpoint_path:
    #     global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    #     saver.restore(sess, ckpt.model_checkpoint_path)
    # print('Loading success, global_step is %s' % global_step)

    #non-transfer
    sess.run(tf.global_variables_initializer())

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        for step in np.arange(MAX_STEP):
            if coord.should_stop():
                break
            _, tra_loss, tra_acc = \
                sess.run([train_op, train_loss, train__acc])
            if step % 50 == 0:
                print('Step %d, train loss = %.2f, train accuracy = %.2f%%' % (step, tra_loss, tra_acc * 100.0))
                summary_str = sess.run(summary_op)
                train_writer.add_summary(summary_str, step)
            if step % 500 == 0 or (step + 1) == MAX_STEP:
                checkpoint_path = os.path.join(logs_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()

    coord.join(threads)
    sess.close()


def evaluate_one_image(test1_data, test2_data, logs_dir, total):
   '''Test one image against the saved models and parameters
   '''
   # you need to change the directories to yours.
   for i in range(0, total):
       with tf.Graph().as_default():
           logitRGB, image_array, train_label, randomList = training.evaluate_one_image(
               test1_data, i, total, N_CLASSES)
           logitD = training_D.evaluate_D_image(test2_data, randomList, i, total, N_CLASSES)
           # fusing 2
           H_Feature = tf.reshape(logitD, [-1])
           L_Feature = tf.reshape(logitRGB, [-1])
           local4 = L_Feature + H_Feature;

           # fusing1
        #    local4 = tf.concat([logitD, logitRGB], 1)
        
           with tf.variable_scope('local3', reuse=None) as scope:
               reshape = tf.reshape(local4, shape=[1, -1])
               dim = reshape.get_shape()[1].value
               weights = tf.get_variable('weights',
                                         shape=[dim, 128],
                                         dtype=tf.float32,
                                         initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
               biases = tf.get_variable('biases',
                                        shape=[128],
                                        dtype=tf.float32,
                                        initializer=tf.constant_initializer(0.1))
               local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)

           with tf.variable_scope('softmax_linear') as scope:
               weights = tf.get_variable('softmax_linear',
                                         shape=[128, 2],
                                         dtype=tf.float32,
                                         initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
               biases = tf.get_variable('biases',
                                        shape=[2],
                                        dtype=tf.float32,
                                        initializer=tf.constant_initializer(0.1))
               train_logits = tf.add(tf.matmul(local3, weights), biases, name='softmax_linear')
           logit = tf.nn.softmax(train_logits)
           x = tf.placeholder(tf.float32, shape=[32, 32, 3])
           saver = tf.train.Saver()

           with tf.Session() as sess:
               print("Reading checkpoints...")
               ckpt = tf.train.get_checkpoint_state(logs_dir)
               if ckpt and ckpt.model_checkpoint_path:
                   global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                   saver.restore(sess, ckpt.model_checkpoint_path)
                   print('Loading success, global_step is %s' % global_step)
               else:
                   print('No checkpoint file found')
               prediction = sess.run(logit, feed_dict={x: image_array})
               max_index = np.argmax(prediction)
               if max_index == 0:
                   print('This is a baseballDiamond with possibility %.6f' % prediction[:, 0],train_label[i])
               else:
                   print('This is a golf with possibility %.6f' % prediction[:, 1],train_label[i])


if __name__ == "__main__":
    N_CLASSES = 2
    # resize the image, if the input image is too large, training will be very slow.
    IMG_W = 32
    IMG_H = 32
    BATCH_SIZE = 16
    CAPACITY = 1000
    MAX_STEP = 1000  # with current parameters, it is suggested to use MAX_STEP>10k
    learning_rate = 0.0001
    total = 9
    total = total - 1#represents the total number of datasets, and index start on -1.
    logs_dir = 'output/log2.4_01/'
    model1_data = 'data/train_m1/'
    model2_data = 'data/train_m2/'
    test1_data = 'data/test_m1/'
    test2_data = 'data/test_m2/'
    # define the target
    target = 'test'

    if target == 'train':
        classifier(N_CLASSES, IMG_W, IMG_H, BATCH_SIZE, MAX_STEP, CAPACITY,
                model1_data, model2_data, learning_rate, logs_dir, total)
    else:
        evaluate_one_image(test1_data, test2_data, logs_dir, total)

