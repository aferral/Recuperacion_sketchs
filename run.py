import numpy as np

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


# Leer tf record en dataset y comprobar que imagenes sigan bien (y que esten bien mezcladas)


train_tf_record = "/home/aferral/Escritorio/temp/train_2018_07_26__21_50_54.tfrecord"



# Consigue dataset
def get_dataset(tf_records,epochs,batch_size,n_classes,shuffle_buffer=500):
  
  def parse_function(example_proto):
    features = {"image_raw": tf.FixedLenFeature((), tf.string),
                "label": tf.FixedLenFeature((), tf.int64)
                }
    parsed_features = tf.parse_single_example(example_proto, features)
    flat_image = tf.decode_raw(parsed_features["image_raw"], tf.uint8)
    
    reconst = tf.cast(tf.reshape(flat_image, (128,128)) ,tf.float32,name='reconstructed_image')

    return reconst, parsed_features["label"]

  dataset = tf.data.TFRecordDataset(tf_records).map(parse_function)
  
  # preprocesss .map(preprocess)
  dataset = dataset.shuffle(shuffle_buffer).repeat(epochs).batch(batch_size).cache()
 
  return dataset

from tensorflow.contrib.layers import fully_connected, conv2d,max_pool2d
from tensorflow.contrib.layers import xavier_initializer, xavier_initializer_conv2d
from tensorflow.layers import batch_normalization

# crear dataset train, test
def arch_skNetI(bx,by):
  
  conv_initializer = xavier_initializer_conv2d()
  dense_initializer = xavier_initializer()
  
  net = tf.reshape(bx,[-1,128,128,1],name="input")
  bn = lambda x : batch_normalization(x)
  
  
  #conv1_1[64]
  #conv1_2[64]
  #maxpool1
  net=conv2d(net,64,3,stride=1,padding='SAME', activation_fn=tf.nn.relu,weights_initializer=conv_initializer,weights_regularizer=None,biases_initializer=tf.zeros_initializer())
  net=bn(net)
  net=conv2d(net,64,3,stride=1,padding='SAME', activation_fn=tf.nn.relu,weights_initializer=conv_initializer,weights_regularizer=None,biases_initializer=tf.zeros_initializer())
  net=bn(net)
  net=max_pool2d(net,3,stride=2)

  #conv2_1[128], 
  #conv2_2[128],
  #maxpool2,
  net=conv2d(net,128,3,stride=1,padding='SAME', activation_fn=tf.nn.relu,weights_initializer=conv_initializer,weights_regularizer=None,biases_initializer=tf.zeros_initializer(),)
  net=bn(net)
  net=conv2d(net,128,3,stride=1,padding='SAME', activation_fn=tf.nn.relu,weights_initializer=conv_initializer,weights_regularizer=None,biases_initializer=tf.zeros_initializer(),)
  net=bn(net)
  net=max_pool2d(net,3,stride=2)


  #conv3_1[128], 
  #conv3_2[128], 
  #maxpool3,
  net=conv2d(net,128,3,stride=1,padding='SAME', activation_fn=tf.nn.relu,weights_initializer=conv_initializer,weights_regularizer=None,biases_initializer=tf.zeros_initializer(),)
  net=bn(net)
  net=conv2d(net,128,3,stride=1,padding='SAME', activation_fn=tf.nn.relu,weights_initializer=conv_initializer,weights_regularizer=None,biases_initializer=tf.zeros_initializer(),)
  net=bn(net)
  net=max_pool2d(net,3,stride=2)


  #conv4_1[256],
  #conv4_2[256],
  #maxpool3,
  net=conv2d(net,256,3,stride=1,padding='SAME', activation_fn=tf.nn.relu,weights_initializer=conv_initializer,weights_regularizer=None,biases_initializer=tf.zeros_initializer(),)
  net=bn(net)
  net=conv2d(net,256,3,stride=1,padding='SAME', activation_fn=tf.nn.relu,weights_initializer=conv_initializer,weights_regularizer=None,biases_initializer=tf.zeros_initializer(),)
  net=bn(net)
  net=max_pool2d(net,3,stride=2)
  
  net = tf.contrib.layers.flatten(net)


  #fc_1[1024],
  net=fully_connected(net,1024,activation_fn=tf.nn.relu,
      weights_initializer=dense_initializer,
      weights_regularizer=None,
      biases_initializer=tf.zeros_initializer())
  # fc_2[100] output
  net=fully_connected(net,100,activation_fn=None,
      weights_initializer=dense_initializer,
      weights_regularizer=None,
      biases_initializer=tf.zeros_initializer())
  return net




bs = 200
lr=0.001
dataset=get_dataset(train_tf_record,1,bs,100,shuffle_buffer=100000)




# Armar arquitectura

iterator = tf.data.Iterator.from_structure(dataset.output_types,dataset.output_shapes)

bx,by = iterator.get_next()
logits = arch_skNetI(bx,by)

pred = tf.nn.softmax(logits, name='prediction')

loss = tf.losses.sparse_softmax_cross_entropy(by, logits)
train_step = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

# get accuracy
prediction = tf.argmax(logits, 1)
equality = tf.equal(prediction, by)
accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))
              

def zip_folder(f_path,out_path_zip):

    fantasy_zip = zipfile.ZipFile(out_path_zip, 'w')

    for folder, subfolders, files in os.walk(f_path):
        for file in files:
            fantasy_zip.write(os.path.join(folder, file),
                              os.path.relpath(os.path.join(folder, file)),
                              compress_type=zipfile.ZIP_DEFLATED)
    fantasy_zip.close()
    pass
  

def save_model(saver,sess,save_name,upload=True):
  now = datetime.now().strftime("%Y_%m_%d__%H:%M:%S")
  path_model_checkpoint = os.path.join('saved_models',save_name,now)
  print("Saving model at {0}".format(path_model_checkpoint))
  os.makedirs(path_model_checkpoint,exist_ok=True)
  path_checkpoint = os.path.join(path_model_checkpoint,'saved_model')
  saver.save(sess, path_checkpoint)
  
  # upload to drive
  if upload:
    zip_name = 'ziped_{0}_{1}.zip'.format(save_name,now)
    out_p =os.path.join('saved_models',zip_name)
    zip_folder(path_model_checkpoint,out_p)
    save_to_drive(out_p,zip_name,"tarea_deep")
  



with tf.Session() as sess:

  init_op = tf.global_variables_initializer()
  sess.run(init_op)
  
  sess.run(iterator.make_initializer(dataset))

  
  saver = tf.train.Saver()


  i = 0

  # batch dataset 100.000 / 200 = 500 batchs
  
  while True:
      try:
          if i % 10 == 0:
            now=datetime.now().strftime("%Y_%m_%d %H:%M:%S")
            l, _, acc = sess.run([loss, train_step, accuracy])
            print("t: {} , It: {}, loss_batch: {:.3f}, batch_accuracy: {:.2f}%".format(now,i, l, acc * 100))
          else:
            
            sess.run(train_step)


          if i % 50 == 0:
            save_model(saver,sess,'train_1',upload=False)

          i += 1

      except tf.errors.OutOfRangeError:
          print('break at {0}'.format(i))
          break


  # todo TEST SET EVAL



  # Save model
  save_model(saver,sess,'train_1',upload=False)

  
 
#   print(a[0]) # Esto revisa que tan ordenadas estan las clases




# arquitecgura en funcion entrenar - checkpoints
# opcion partir de check point

# Usar loop de train, test.

# evaluar accuracy sklearn.
# crear reporte csv o algo asi.

# paso 2 activaciones.
# cargar modelo 
# cargar dataset a transformar. (test)
# pasar nombre capas a extraer
# guardar activaciones???
# loop search 1 vs otros.
# precomputar distancias en vectores.
# conseguir ranking.
# calcular map


