import os
import numpy as np
import tensorflow as tf

from config import img_resize_method
from head import debug, weights, self.kernel_3x3, self.kernel_5x5

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


class CNN():

    # to be updated
    def __init__(self):
        # Untrainable Gaussian blur kernel
        gaussin_blur_3x3 = np.divide([
            [1., 2., 1.],
            [2., 4., 2.],
            [1., 2., 1.],
        ], 16.) # (3, 3)
        gaussin_blur_3x3 = np.stack((gaussin_blur_3x3, gaussin_blur_3x3), axis=-1) # (3, 3, 2)
        gaussin_blur_3x3 = np.stack((gaussin_blur_3x3, gaussin_blur_3x3), axis=-1) # (3, 3, 2, 2)

        gaussin_blur_5x5 = np.divide([
            [1.,  4.,  7.,  4., 1.],
            [4., 16., 26., 16., 4.],
            [7., 26., 41., 26., 7.],
            [4., 16., 26., 16., 4.],
            [1.,  4.,  7.,  4., 1.],
        ], 273.) # (5, 5)

        gaussin_blur_5x5 = np.stack((gaussin_blur_5x5, gaussin_blur_5x5), axis=-1) # (5, 5, 2)
        gaussin_blur_5x5 = np.stack((gaussin_blur_5x5, gaussin_blur_5x5), axis=-1) # (5, 5, 2, 2)

        self.self.kernel_3x3 = tf.Variable(tf.convert_to_tensor(gaussin_blur_3x3, dtype=tf.float32), trainable=False)
        self.self.kernel_5x5 = tf.Variable(tf.convert_to_tensor(gaussin_blur_3x3, dtype=tf.float32), trainable=False)


        # Trainable weights for each layer
        self.weights = {
            'conv0': tf.Variable(tf.truncated_normal([3, 1, -1, 32], stddev=0.01), trainable=True),
            'conv1': tf.Variable(tf.truncated_normal([3, 1, 32, 64], stddev=0.01), trainable=True),
            'conv2': tf.Variable(tf.truncated_normal([3, 1, 64, 128], stddev=0.01), trainable=True)
        }
    
    def get_weight(scope):
        return weights[scope]

    # to be updated
    def get_loss(predict, real):
        
        if debug:
            assert predict.get_shape().as_list()[1:] == [224, 224, 2]
            assert real.get_shape().as_list()[1:] == [224, 224, 2]

        predict_blur_3x3 = tf.nn.conv2d(predict, self.kernel_3x3, strides=[1,1,1,1], padding='SAME', name="predict_blur_3x3")
        predict_blur_5x5 = tf.nn.conv2d(predict, self.self.kernel_5x5, strides=[1,1,1,1], padding='SAME',name="predict_blur_5x5")

        real_blur_3x3 = tf.nn.conv2d(predict, self.kernel_3x3, strides=[1,1,1,1], padding='SAME', name="real_blur_3x3")
        real_blur_5x5 = tf.nn.conv2d(predict, self.kernel_5x5, strides=[1,1,1,1], padding='SAME',name="real_blur_5x5")

        diff_original = tf.reduce_sum(tf.squared_difference(predict, real), name="diff_original")
        diff_blur_3x3 = tf.reduce_sum(tf.squared_difference(predict_blur_3x3, real_blur_3x3), name="diff_blur_3x3")
        diff_blur_5x5 = tf.reduce_sum(tf.squared_difference(predict_blur_5x5, real_blur_5x5), name="diff_blur_5x5")

        return (diff_original + diff_blur_3x3 + diff_blur_5x5) / 3


    def batch_normal(input, scope, is_training):
        return tf.layers.batch_normalization(input, training=is_training, name=scope)


    def pool_layer(self, input, name):
        return tf.nn.max_pool(input, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name=name)

    def conv_layer(self, input, scope, is_training, relu=True, is_batchNorm=True):
        weight = self.get_weight(scope)
        with tf.variable_scope(scope):
            conv = tf.nn.conv2d(input, weight, strides=[1,1,1,1], padding='SAME', name="conv"+"scope")
            if is_batchNorm is True:
                conv = self.batch_normal(conv, scope+"_batchNorm", is_training)
            if relu:
                conv = tf.nn.relu(conv, name="relu")
            else:
                conv = tf.nn.sigmoid(conv, name="sigmoid")
        return conv

    def fc_layer(self, input, name):
        weight = self.get_weight(name)
        with tf.variable_scope(name):
            shape = input.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d

            x = tf.reshape(input, [-1, dim])
            weights = self.get_weight(name + "_weight")
            biases = self.get_weight(name + "bias")

            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def build(self, input, vgg, is_training):
        
        if debug:
            print("input's shape:", input.get_shape().as_list())

        conv0 = self.conv_layer(input, "conv0", is_training, is_batchNorm=False)
        pool0 = self.pool_layer(conv0, "pool0")
        batch0 = self.batch_normal(pool0, "batch0", is_training)
        
        if debug:
            print("conv0's shape:", conv0.get_shape().as_list())
            print("pool0's shape:", pool0.get_shape().as_list())
        
        conv1 = self.conv_layer(input, "conv1", is_training, is_batchNorm=False)
        pool1 = self.pool_layer(conv1, "pool1")
        batch1 = self.batch_normal(pool1, "batch1", is_training)
        
        if debug:
            print("conv1's shape:", conv1.get_shape().as_list())
            print("pool1's shape:", pool1.get_shape().as_list())

        conv2 = self.conv_layer(input, "conv2", is_training, is_batchNorm=False)
        pool2 = self.pool_layer(conv2, "pool2")
        batch2 = self.batch_normal(pool2, "batch2", is_training)
        
        if debug:
            print("conv2's shape:", conv2.get_shape().as_list())
            print("pool2's shape:", pool2.get_shape().as_list())

       flat = tf.layers.flatten(batch2, "flatten")

       if debug:
           assert len(flat.get_shape().as_list()) == 2
        
        output = tf.layers.dense(
            flat, 
            unit = 12, 
            activation = tf.nn.relu
        )

        if debug:
            assert len(output.get_shape().as_list()) == 1

        return output    
