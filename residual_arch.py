import numpy as np


from tensorflow.layers import dense, conv2d,max_pooling2d
from tensorflow.contrib.layers import xavier_initializer, xavier_initializer_conv2d
from tensorflow.layers import batch_normalization


import matplotlib.pyplot as plt
import pandas as pd
import io
import random
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from subprocess import call
import zipfile


# crear dataset train, test
def arch_skNetI_residual(bx, by):
    conv_initializer = xavier_initializer_conv2d()
    dense_initializer = xavier_initializer()

    net = tf.reshape(bx, [-1, 128, 128, 1], name="input")
    bn = lambda x: batch_normalization(x)

    # conv1_1[64]
    # conv1_2[64]
    # maxpool1
    net = conv2d(net, 64, 3, strides=(1, 1), padding='same', activation=tf.nn.relu, kernel_initializer=conv_initializer,
                 kernel_regularizer=None, bias_initializer=tf.zeros_initializer(), name='conv1_1')
    net = bn(net)
    net = conv2d(net, 64, 3, strides=(1, 1), padding='same', activation=tf.nn.relu, kernel_initializer=conv_initializer,
                 kernel_regularizer=None, bias_initializer=tf.zeros_initializer(), name='conv1_2')
    net = bn(net)
    net = max_pooling2d(net, 3, 2, name='maxpool_1')
    maxpool_1 = net

    # conv2_1[64],
    # conv2_2[64],
    # residual_1=conv2_2 + maxpool_1,
    net = conv2d(net, 64, 3, strides=(1, 1), padding='same', activation=tf.nn.relu, kernel_initializer=conv_initializer,
                 kernel_regularizer=None, bias_initializer=tf.zeros_initializer(), name='conv2_1')
    net = bn(net)
    net = conv2d(net, 64, 3, strides=(1, 1), padding='same', activation=tf.nn.relu, kernel_initializer=conv_initializer,
                 kernel_regularizer=None, bias_initializer=tf.zeros_initializer(), name='conv2_2')
    net = bn(net)
    net = net + maxpool_1  # max_pool2d(net,3,stride=2,name='maxpool_2')
    residual_1 = net

    # conv3_1[64],
    # conv3_2[64],
    # residual_2=conv3_2+residual_1,
    net = conv2d(net, 64, 3, strides=(1, 1), padding='same', activation=tf.nn.relu, kernel_initializer=conv_initializer,
                 kernel_regularizer=None, bias_initializer=tf.zeros_initializer(), name='conv3_1')
    net = bn(net)
    net = conv2d(net, 64, 3, strides=(1, 1), padding='same', activation=tf.nn.relu, kernel_initializer=conv_initializer,
                 kernel_regularizer=None, bias_initializer=tf.zeros_initializer(), name='conv3_2')
    net = bn(net)
    net = net + residual_1  # max_pool2d(net,3,stride=2)
    residual_2 = net

    # conv4_1[128],
    # maxpool3,
    net = conv2d(net, 128, 3, strides=(1, 1), padding='same', activation=tf.nn.relu,
                 kernel_initializer=conv_initializer, kernel_regularizer=None, bias_initializer=tf.zeros_initializer(),
                 name='conv4_1')
    net = bn(net)
    # net=conv2d(net,256,3,stride=(1,1),padding='SAME', activation_fn=tf.nn.relu,kernel_initializer=conv_initializer,kernel_regularizer=None,bias_initializer=tf.zeros_initializer(),)
    # net=bn(net)
    net = max_pooling2d(net, 3, 2, name='maxpool_2')
    maxpool_2 = net

    # conv5_1[128],
    # conv5_2[128],
    # residual_3=conv5_2+maxpool_2,
    net = conv2d(net, 128, 3, strides=(1, 1), padding='same', activation=tf.nn.relu,
                 kernel_initializer=conv_initializer, kernel_regularizer=None, bias_initializer=tf.zeros_initializer(),
                 name='conv5_1')
    net = bn(net)
    net = conv2d(net, 128, 3, strides=(1, 1), padding='same', activation=tf.nn.relu,
                 kernel_initializer=conv_initializer, kernel_regularizer=None, bias_initializer=tf.zeros_initializer(),
                 name='conv5_2')
    net = bn(net)
    net = net + maxpool_2  # max_pool2d(net,3,stride=2)
    residual_3 = net

    # conv6_1[128],
    # conv6_2[128],
    # residual_4=conv5_2+residual_3,
    net = conv2d(net, 128, 3, strides=(1, 1), padding='same', activation=tf.nn.relu,
                 kernel_initializer=conv_initializer, kernel_regularizer=None, bias_initializer=tf.zeros_initializer(),
                 name='conv6_1')
    net = bn(net)
    net = conv2d(net, 128, 3, strides=(1, 1), padding='same', activation=tf.nn.relu,
                 kernel_initializer=conv_initializer, kernel_regularizer=None, bias_initializer=tf.zeros_initializer(),
                 name='conv6_2')
    net = bn(net)
    net = net + residual_3  # max_pool2d(net,3,stride=2)
    residual_4 = net

    # conv7_1[256],
    # maxpool_3
    net = conv2d(net, 256, 3, strides=(1, 1), padding='same', activation=tf.nn.relu,
                 kernel_initializer=conv_initializer, kernel_regularizer=None, bias_initializer=tf.zeros_initializer(),
                 name='conv7_1')
    net = bn(net)
    # net=conv2d(net,128,3,stride=(1,1),padding='SAME', activation_fn=tf.nn.relu,kernel_initializer=conv_initializer,kernel_regularizer=None,bias_initializer=tf.zeros_initializer(),name='conv6_2')
    # net=bn(net)
    net = max_pooling2d(net, 3, 2, name='maxpool_3')
    maxpool_3 = net

    # conv8_1[256],
    # conv8_2[256],
    # residual_5=conv8_2+maxpool_3,
    net = conv2d(net, 256, 3, strides=(1, 1), padding='same', activation=tf.nn.relu,
                 kernel_initializer=conv_initializer, kernel_regularizer=None, bias_initializer=tf.zeros_initializer(),
                 name='conv8_1')
    net = bn(net)
    net = conv2d(net, 256, 3, strides=(1, 1), padding='same', activation=tf.nn.relu,
                 kernel_initializer=conv_initializer, kernel_regularizer=None, bias_initializer=tf.zeros_initializer(),
                 name='conv8_2')
    net = bn(net)
    net = net + maxpool_3  # max_pool2d(net,3,stride=2)
    residual_5 = net

    # conv9_1[256],
    # conv9_2[256],
    # residual_6=conv9_2+residual_5,
    net = conv2d(net, 256, 3, strides=(1, 1), padding='same', activation=tf.nn.relu,
                 kernel_initializer=conv_initializer, kernel_regularizer=None, bias_initializer=tf.zeros_initializer(),
                 name='conv9_1')
    net = bn(net)
    net = conv2d(net, 256, 3, strides=(1, 1), padding='same', activation=tf.nn.relu,
                 kernel_initializer=conv_initializer, kernel_regularizer=None, bias_initializer=tf.zeros_initializer(),
                 name='conv9_2')
    net = bn(net)
    net = net + residual_5  # max_pool2d(net,3,stride=2)
    residual_4 = net

    # conv10_1[256],
    # maxpool_4
    net = conv2d(net, 256, 3, strides=(1, 1), padding='same', activation=tf.nn.relu,
                 kernel_initializer=conv_initializer, kernel_regularizer=None, bias_initializer=tf.zeros_initializer(),
                 name='conv10_1')
    net = bn(net)
    # net=conv2d(net,128,3,stride=(1,1),padding='SAME', activation_fn=tf.nn.relu,kernel_initializer=conv_initializer,kernel_regularizer=None,bias_initializer=tf.zeros_initializer(),name='conv6_2')
    # net=bn(net)
    net = max_pooling2d(net, 3, 2, name='maxpool_4')

    net = tf.contrib.layers.flatten(net)

    # fc_1[1024],
    net = dense(net, 1024, activation=tf.nn.relu,
                kernel_initializer=dense_initializer,
                kernel_regularizer=None,
                bias_initializer=tf.zeros_initializer(), name='fc_1')
    # fc_2[100] output
    net = dense(net, 100, activation=None,
                kernel_initializer=dense_initializer,
                kernel_regularizer=None,
                bias_initializer=tf.zeros_initializer(), name='fc_2')
    return net

