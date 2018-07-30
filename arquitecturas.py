from tensorflow.layers import conv2d
from tensorflow.contrib.layers import xavier_initializer, xavier_initializer_conv2d, max_pool2d, fully_connected
from tensorflow.layers import batch_normalization
import tensorflow as tf



def redSimple(input):
    out = tf.reshape(input, [-1, 128, 128, 1])
    out = conv2d(out, 64, 3, activation_fn=tf.nn.relu,weights_initializer=xavier_initializer_conv2d())
    out = batch_normalization(out)
    
    out = conv2d(out, 64, 3, activation_fn=tf.nn.relu,weights_initializer=xavier_initializer_conv2d())
    out = batch_normalization(out)
    out = max_pool2d(out, 3, stride=2)
    out = conv2d(out, 128, 3,activation_fn=tf.nn.relu, weights_initializer=xavier_initializer_conv2d())
    out = batch_normalization(out)
    out = conv2d(out, 128, 3,activation_fn=tf.nn.relu, weights_initializer=xavier_initializer_conv2d())
    out = batch_normalization(out)
    out = max_pool2d(out, 3, stride=2)


    out = conv2d(out, 128, 3,activation_fn=tf.nn.relu, weights_initializer=xavier_initializer_conv2d())
    out = batch_normalization(out)
    out = conv2d(out, 128, 3,activation_fn=tf.nn.relu, weights_initializer=xavier_initializer_conv2d())
    out = batch_normalization(out)
    out = max_pool2d(out, 3, stride=2)


    out = conv2d(out, 256, 3,activation_fn=tf.nn.relu, weights_initializer=xavier_initializer_conv2d())
    out = batch_normalization(out)
    out = conv2d(out, 256, 3,activation_fn=tf.nn.relu, weights_initializer=xavier_initializer_conv2d())
    out = batch_normalization(out)
    out = max_pool2d(out, 3, stride=2)

    out = tf.contrib.layers.flatten(out)

    out = fully_connected(out, 1024, activation_fn=tf.nn.relu,weights_initializer=xavier_initializer())
    out = fully_connected(out, 100, activation_fn=None, weights_initializer=xavier_initializer())
    return out


def redResidua(input):
    out = tf.reshape(input, [-1, 128, 128, 1])


    out = conv2d(out, 64, 3, padding='same', activation=tf.nn.relu, kernel_initializer=xavier_initializer_conv2d())
    out = batch_normalization(out)
    out = conv2d(out, 64, 3, padding='same', activation=tf.nn.relu, kernel_initializer=xavier_initializer_conv2d())
    out = batch_normalization(out)
    out = max_pool2d(out, 3, stride=2)
    max1 = out


    out = conv2d(out, 64, 3, padding='same', activation=tf.nn.relu, kernel_initializer=xavier_initializer_conv2d())
    out = batch_normalization(out)
    out = conv2d(out, 64, 3, padding='same', activation=tf.nn.relu, kernel_initializer=xavier_initializer_conv2d())
    out = batch_normalization(out)
    out = out + max1
    r1 = out


    out = conv2d(out, 64, 3, padding='same', activation=tf.nn.relu, kernel_initializer=xavier_initializer_conv2d())
    out = batch_normalization(out)
    out = conv2d(out, 64, 3, padding='same', activation=tf.nn.relu, kernel_initializer=xavier_initializer_conv2d())
    out = batch_normalization(out)
    out = out + r1


    out = conv2d(out, 128, 3, padding='same', activation=tf.nn.relu,kernel_initializer=xavier_initializer_conv2d())
    out = batch_normalization(out)

    out = max_pool2d(out, 3, stride=2)
    max2 = out


    out = conv2d(out, 128, 3, padding='same', activation=tf.nn.relu,kernel_initializer=xavier_initializer_conv2d())
    out = batch_normalization(out)
    out = conv2d(out, 128, 3, padding='same', activation=tf.nn.relu,kernel_initializer=xavier_initializer_conv2d())
    out = batch_normalization(out)
    out = out + max2
    r3 = out


    out = conv2d(out, 128, 3, padding='same', activation=tf.nn.relu,kernel_initializer=xavier_initializer_conv2d())
    out = batch_normalization(out)
    out = conv2d(out, 128, 3, padding='same', activation=tf.nn.relu,kernel_initializer=xavier_initializer_conv2d())
    out = batch_normalization(out)
    out = out + r3


    out = conv2d(out, 256, 3, padding='same', activation=tf.nn.relu,kernel_initializer=xavier_initializer_conv2d())
    out = batch_normalization(out)
    out = max_pool2d(out, 3, stride=2)
    max3 = out


    out = conv2d(out, 256, 3, padding='same', activation=tf.nn.relu,kernel_initializer=xavier_initializer_conv2d())
    out = batch_normalization(out)
    out = conv2d(out, 256, 3, padding='same', activation=tf.nn.relu,kernel_initializer=xavier_initializer_conv2d())
    out = batch_normalization(out)
    out = out + max3
    r5 = out


    out = conv2d(out, 256, 3, padding='same', activation=tf.nn.relu,kernel_initializer=xavier_initializer_conv2d())
    out = batch_normalization(out)
    out = conv2d(out, 256, 3, padding='same', activation=tf.nn.relu,kernel_initializer=xavier_initializer_conv2d())
    out = batch_normalization(out)
    out = out + r5

    out = conv2d(out, 256, 3, padding='same', activation=tf.nn.relu,kernel_initializer=xavier_initializer_conv2d())
    out = batch_normalization(out)
    out = max_pool2d(out, 3, stride=2)

    out = tf.contrib.layers.flatten(out)

    out = fully_connected(out, 1024, activation_fn=tf.nn.relu,weights_initializer=xavier_initializer())
    out = fully_connected(out, 100, activation_fn=None, weights_initializer=xavier_initializer())
    return out

